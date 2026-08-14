"""``EgoStitchModel``: the three-component composite (design 2026-08-02 §3-§7).

Wires `NeighborhoodGenerator` -> `GraphEncoder` -> `PairClassifier` into the
single module that replaces `EgoStitchE2E`. Every submodule is built through
`src.model.egostitch.registry`'s `build_generator`/`build_encoder`/
`build_classifier` helpers from `cfg.generator`/`cfg.encoder`/
`cfg.classifier`'s own ``name`` field (design §3, §8, §12 P3) -- swapping a
component is a config change, not a code change.

`generator.name: "null"` (`NullGenerator`, `generator/null.py`) is the one
generator registered today that never emits a graph: `stitch` always returns
`None`, so no `PairConditioning` is ever built and `self.encoder` is not
constructed at all (§3.4's ``encoder: GraphEncoder | None``) -- a
null-generator model's `state_dict` therefore carries no generator or
encoder parameters, and its classifier runs exactly as unconditioned as the
`f_logit` hard bypass (design §3.3, §12 P3 acceptance criterion 5).

Behaviour-preserving by construction for the default (`egostitch_imagine` +
`ste_typed` + `b0_v31`) configuration: every method here reproduces what
`EgoStitchE2E` computed, just re-expressed through the three components'
public contracts (`generator.encode_node`/`.stitch`, `encoder.forward`,
`classifier.forward`). `tests/model/test_egostitch_e2e_model.py` (retargeted
onto this module) and `tests/model/test_composite.py` are the proof.

Two numerics-critical details this module must reproduce, not merely mimic:

1. **The self-row encoding optimization.** `_pair_node_states` (mirroring
   `e2e_model.py:223-253`) calls `generator.encode_node` for endpoint A on
   every row but for endpoint B only on non-self rows, then merges A's state
   into the self rows. The batch factory guarantees ``x_a == x_b`` on self
   rows, so the merge is exact, not an approximation -- and it halves
   generator cost on `probe_states`' self-pair batches.
2. **`E2ENodeState.encoded` holds already-encoded token states, cached once
   per node.** `B0V31PairClassifier` is itself split into a cacheable
   `encode_tokens` phase and a pair-level `forward` phase (correction,
   2026-08-03, mirroring `NeighborhoodGenerator.encode_node`/`.stitch`): a
   `forward` that ran `SiameseEncoder` on raw tokens internally, every call,
   would silently defeat the per-node cache `score_universe.py`'s
   ``node_cache`` depends on -- a node scored against many counterparts would
   be siamese-encoded once per pair instead of once, total. So
   `encode_node_state` calls `classifier.encode_tokens(emb, length)` (not a
   raw pass-through), and `score_pair_context` hands the cached, already-
   encoded state to the classifier as `PairInputs.tokens_a`/`tokens_b` --
   never calling `classifier.encode_tokens` a second time there.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, NamedTuple, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from src.data.feature_stats import FeatureStats
from src.model.egostitch.classifier.b0_v31 import (
    NULL_ALL_HEAD,
    NULL_NONE,
    masks_for_null,
)
from src.model.egostitch.classifier.base import HeadNullMasks, PairClassifier
from src.model.egostitch.config import E2EConfig, EgoStitchConfig
from src.model.egostitch.encoder.base import GraphEncoder
from src.model.egostitch.generator import (
    EgoStitchImagineGenerator,
    GeneratorNodeState,
    NullGenerator,
)
from src.model.egostitch.generator.assemble import ScaffoldInputPerturbation
from src.model.egostitch.generator.base import NeighborhoodGenerator
from src.model.egostitch.generator.imagine import SlotSet
from src.model.egostitch.graph import GraphEmbedding, ImaginedGraph, PairConditioning, PairInputs
from src.model.egostitch.registry import (
    build_classifier,
    build_encoder,
    build_generator,
    resolve_generator_calibration,
)


class E2ENodeState(NamedTuple):
    """Cacheable per-node trunk and imagination state.

    Field names/order are frozen (design amendment 2026-08-03): callers that
    score many pairs over a shared node universe (`score_universe.py`,
    `train_egostitch.py`) construct this NamedTuple directly, field-by-field,
    to reassemble cached per-node state into per-pair batches.

    Attributes:
        encoded: The classifier's own `encode_tokens` output (see module
            docstring point 2) -- `B0V31PairClassifier.forward` consumes this
            directly and never re-applies `SiameseEncoder`, so this field is
            already encoded, not a raw pass-through.
        length: Unpadded token counts for ``encoded``.
        slots: The generator's own `SlotSet` (`GeneratorNodeState.slots`), or
            `None` for a generator whose `encode_node` produces no such state
            (`NullGenerator.encode_node` echoes its input back rather than
            allocating one -- design §3.3, `generator/null.py`).
        projected_x: The generator's own projected features
            (`GeneratorNodeState.projected_x`), or `None` under the same
            condition as `slots`.
        ground_ids: The generator's own grounding ids
            (`GeneratorNodeState.ground_ids`).
    """

    encoded: torch.Tensor
    length: torch.Tensor
    slots: SlotSet | None
    projected_x: torch.Tensor | None
    ground_ids: torch.Tensor | None


class E2EPairContext(NamedTuple):
    """One shared pair context consumed by every hard-bypass head.

    Field names/order are frozen for the same reason as `E2ENodeState`.

    Attributes:
        encoded_a: Endpoint-A already-encoded token state
            (`PairInputs.tokens_a`-shaped).
        encoded_b: Endpoint-B already-encoded token state
            (`PairInputs.tokens_b`-shaped).
        len_a: Endpoint-A token counts.
        len_b: Endpoint-B token counts.
        topo_ab: AB-direction encoder tokens (`GraphEmbedding.tokens`), or
            `None` when topology was not needed.
        topo_ba: BA-direction encoder tokens, or `None`.
        plan: The generator's Sinkhorn plan (``graph.aux["plan"]``), or
            `None`.
        log_plan: The generator's Sinkhorn log-plan
            (``graph.aux["log_plan"]``), or `None`.
        pooled_ab: AB-direction encoder pooled summary
            (`GraphEmbedding.pooled`), or `None`. Trailing and optional so
            every existing positional construction of this NamedTuple (the
            field order is frozen -- see `E2ENodeState`) keeps working
            unchanged. Carried because the conditioning ladder's
            ``film_logit`` and ``pooled_adapter`` rungs read `pooled`, and an
            encoder whose pooled summary is a *learned* readout (`grit_gmt`'s
            PMA seed tokens) cannot be reconstructed from `topo_ab` by a mean
            -- `score_pair_context`'s ``tokens.mean(1)`` fallback would
            silently substitute a different quantity.
        pooled_ba: BA-direction encoder pooled summary, or `None`.
    """

    encoded_a: torch.Tensor
    encoded_b: torch.Tensor
    len_a: torch.Tensor
    len_b: torch.Tensor
    topo_ab: torch.Tensor | None
    topo_ba: torch.Tensor | None
    plan: torch.Tensor | None
    log_plan: torch.Tensor | None
    pooled_ab: torch.Tensor | None = None
    pooled_ba: torch.Tensor | None = None


class EgoStitchModel(nn.Module):
    """The rev-4 three-component topology-conditioned pair encoder.

    Composes `generator` (`NeighborhoodGenerator`), `encoder` (`GraphEncoder`)
    and `classifier` (`PairClassifier`) -- named exactly this, per design §3.4
    -- into the same public surface `EgoStitchE2E` exposed, so
    `train_egostitch.py`/`score_universe.py`/`src/experiments/` retarget
    mechanically.
    """

    def __init__(self, cfg: E2EConfig) -> None:
        """Build every submodule through the registry from `cfg`.

        Args:
            cfg: The nested generator / encoder / classifier hyperparameters
                (design §8): `cfg.generator.name` / `cfg.encoder.name` /
                `cfg.classifier.name` select the concrete implementation via
                `src.model.egostitch.registry`.
        """
        super().__init__()
        self.cfg = cfg
        # Computed once, independent of which generator is selected, so
        # `generator_cfg` (below) stays meaningful -- e.g. for its
        # `.n_ground`/`.slots` fields, read by `src/train_egostitch.py` and
        # `src/experiments/probes.py` regardless of arm -- even for a
        # generator such as `NullGenerator` that carries no `EgoStitchConfig`
        # of its own. Passed into `build_generator` as the *same* object, so
        # identity with `self.generator.cfg` is preserved for a generator
        # that does have one (`registry.resolve_generator_calibration`).
        #
        # `w_rel` is folded in here, not by `resolve_generator_calibration`:
        # it lives on `cfg.encoder` (design §8's pinned schema puts it under
        # `encoder:`, since it gates whether `GraphEncoder` builds a
        # relational head at all), but `EgoStitchConfig.w_rel` is also the
        # *interior loss weight* `losses.py:643`'s `stage1_total`/
        # `_recon_total` multiplies the encoder's own (unweighted) `rel_loss`
        # by -- so `model.generator_cfg.w_rel` must mirror `cfg.encoder.w_rel`
        # for that weighting to be correct, the same cross-component coupling
        # the composite already resolves for `d_model` (task pin, below).
        self._generator_cfg = replace(
            resolve_generator_calibration(cfg.generator), w_rel=cfg.encoder.w_rel
        )
        self.generator: NeighborhoodGenerator[Any, Any, Any] = build_generator(
            cfg.generator, generator_cfg=self._generator_cfg
        )
        self.input_dim = self._generator_cfg.input_dim  # frozen feature dim (spec Sec 0 table)
        self.node_feature_dim = self._generator_cfg.input_dim

        self.encoder: GraphEncoder | None
        if isinstance(self.generator, NullGenerator):
            # Nothing is ever imagined (`NullGenerator.stitch` always returns
            # `None`), so there is nothing to encode: no `GraphEncoder` is
            # constructed at all. Plain `None` here (never registered as a
            # submodule -- `nn.Module.__setattr__` only tracks `Module`
            # instances) is what makes a null-generator `state_dict` carry no
            # encoder parameters (design §12 P3).
            self.encoder = None
        else:
            # Every non-null generator answers for the shape of the graph it
            # emits itself (`NeighborhoodGenerator.graph_dims`), so adding a
            # generator never touches this selection logic. The old inline
            # probe lives on as `EgoStitchImagineGenerator.graph_dims`, which
            # is where its knowledge of that generator's `EgoStitchConfig`
            # belongs.
            feature_dim, num_relations = self.generator.graph_dims()
            self.encoder = build_encoder(
                cfg.encoder,
                in_dim=feature_dim,
                num_relations=num_relations,
                d_model=cfg.classifier.d_model,
            )
        self.classifier: PairClassifier = build_classifier(cfg.classifier, input_dim=self.input_dim)
        if self.encoder is None:
            # No `GraphEncoder` was constructed above (`generator.name:
            # "null"`), so `score_pair_context`/`forward` always clamp
            # `need_topo` off and `cond` is always `None` (module docstring,
            # `_build_pair_context_and_graph`). Any classifier-owned
            # conditioning submodule reachable only through `cond is not
            # None` therefore never runs a forward pass under this
            # composition and would get no gradient -- DDP's
            # `find_unused_parameters=False` raises on the first backward
            # otherwise (CLAUDE.md P1). Freezing here, once, at construction
            # time (not per-caller in `train_egostitch.py`) makes every
            # construction path -- DDP, single-process, `score_universe.py`
            # -- carry the same `requires_grad` state on every rank before
            # any DDP wrap happens.
            self.classifier.freeze_unreachable_conditioning()

    @property
    def generator_cfg(self) -> EgoStitchConfig:
        """The internal generator's `EgoStitchConfig` (today's `EgoStitchE2E.generator_cfg`).

        Preserved verbatim as a pinned attribute (design coordinator
        amendment, 2026-08-03): read directly by
        `src/experiments/probes.py:820,895,985` (`.n_ground`, `.slots`),
        `src/train_egostitch.py:3875` and `src/score_universe.py:1760`.
        Same object identity as `self.generator.cfg` for a generator that has
        one; for a generator that does not (e.g. `NullGenerator`), this is
        still the `cfg.generator`-derived `EgoStitchConfig` computed in
        `__init__`, so `.n_ground`/`.slots` reads stay well-defined.
        """
        return self._generator_cfg

    def set_feature_stats(self, stats: FeatureStats) -> None:
        """Pin the registered F0 standardization constants on the generator.

        A no-op for a generator with nothing to standardize (`NullGenerator`
        does not implement `set_feature_stats`: there are no features it
        ever reads).

        Args:
            stats: The V_fit statistics bundle.
        """
        if isinstance(self.generator, EgoStitchImagineGenerator):
            self.generator.set_feature_stats(stats)

    @property
    def feature_stats_digest_hex(self) -> str:
        """The generator's registered `feature_stats_sha256`, or ``""``.

        Always ``""`` for a generator with no such digest (`NullGenerator`),
        the same value `EgoStitchImagineGenerator.feature_stats_digest_hex`
        itself returns for a stateless standardization mode.
        """
        if isinstance(self.generator, EgoStitchImagineGenerator):
            return self.generator.feature_stats_digest_hex
        return ""

    # ------------------------------------------------------------------ per-node state

    @staticmethod
    def _generator_state(state: E2ENodeState) -> GeneratorNodeState | None:
        """Extract the generator-owned fields of one `E2ENodeState`.

        Returns `None` when `state` carries no such fields (`slots`/
        `projected_x` are `None` together, always the case for a generator
        whose `encode_node` produced no `GeneratorNodeState` -- see
        `encode_node_state`). Consumed only by `self.generator.stitch`; the
        one generator that ever sees a `None` here (`NullGenerator.stitch`)
        ignores its `state_a`/`state_b` arguments unconditionally, so this is
        safe.
        """
        if state.slots is None or state.projected_x is None:
            return None
        return GeneratorNodeState(
            slots=state.slots, projected_x=state.projected_x, ground_ids=state.ground_ids
        )

    @staticmethod
    def _merge_slots(
        base: SlotSet | None, replacement: SlotSet | None, rows: torch.Tensor
    ) -> SlotSet | None:
        """Replace selected batch rows while preserving autograd to both inputs.

        `None` in, `None` out: a generator that never produces a `SlotSet`
        (`NullGenerator`) has nothing to merge.
        """
        if base is None and replacement is None:
            return None
        assert base is not None and replacement is not None, (
            "a generator's encode_node must consistently produce SlotSet or "
            "consistently produce none across every call"
        )
        return SlotSet(*(a.index_copy(0, rows, b) for a, b in zip(base, replacement, strict=True)))

    def encode_node_state(
        self,
        emb: torch.Tensor,
        length: torch.Tensor,
        x: torch.Tensor,
        ground: torch.Tensor,
        ground_ids: torch.Tensor | None = None,
        node_rows: torch.Tensor | None = None,
    ) -> E2ENodeState:
        """Run the cacheable per-node generator and classifier-token-encoder passes.

        Both the generator's `encode_node` and the classifier's
        `encode_tokens` are cacheable per-node phases (module docstring point
        2), so both run here, once per unique node -- not the pair-level
        `stitch`/`forward` phases, which run once per pair.

        `self.generator.encode_node` is called unconditionally (protocol
        compliance, every generator implements it), but its return value is
        only unpacked into `slots`/`projected_x` when it is actually a
        `GeneratorNodeState` -- `NullGenerator.encode_node` instead echoes
        `x` back, having nothing to cache structurally (design §3.3,
        `generator/null.py`), so `E2ENodeState.slots`/`.projected_x` stay
        `None` in that case.

        `node_rows` (context-local F0 row ids) is forwarded to the generator
        when the caller has it and `None` when it does not. Passing it is the
        caller's choice; *needing* it is the generator's contract
        (`OracleStructGenerator` raises without it, the feature-driven
        generators ignore it).
        """
        generated = self.generator.encode_node(x, ground, ground_ids, node_rows=node_rows)
        if isinstance(generated, GeneratorNodeState):
            return E2ENodeState(
                encoded=self.classifier.encode_tokens(emb, length),
                length=length,
                slots=generated.slots,
                projected_x=generated.projected_x,
                ground_ids=generated.ground_ids,
            )
        return E2ENodeState(
            encoded=self.classifier.encode_tokens(emb, length),
            length=length,
            slots=None,
            projected_x=None,
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
        if base.projected_x is None or replacement.projected_x is None:
            projected_x = None
        else:
            projected_x = base.projected_x.index_copy(0, rows, replacement.projected_x)
        return E2ENodeState(
            encoded=encoded_base.index_copy(0, rows, encoded_replacement),
            length=base.length.index_copy(0, rows, replacement.length),
            slots=self._merge_slots(base.slots, replacement.slots, rows),
            projected_x=projected_x,
            ground_ids=ground_ids,
        )

    def _pair_node_states(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[E2ENodeState, E2ENodeState, torch.Tensor]:
        """Encode pair endpoints, using exactly one `generator.encode_node` per self row.

        Reproduces `e2e_model.py:223-253`'s self-row encoding optimization
        (module docstring point 1): endpoint A is encoded for every row,
        endpoint B only for non-self rows, and A's generator state is merged
        into the self rows -- safe because the batch factory guarantees
        ``x_a == x_b`` there.

        Endpoint identity, when the batch carries it, rides along under
        ``node_row_i``/``node_row_j`` (edge batches) or ``node_rows``
        (validation node batches, where both endpoints are the same node).
        Absent keys become `None`; a generator that needs identity says so
        itself rather than having the composite guess.
        """
        missing = {"ground_a", "ground_b"} - batch.keys()
        if missing:
            raise ValueError(f"EgoStitchModel requires real grounding tensors: {sorted(missing)}")
        batch_size = batch["emb_a"].size(0)
        is_self = batch.get(
            "is_self", torch.zeros(batch_size, dtype=torch.bool, device=batch["emb_a"].device)
        )
        shared_rows = batch.get("node_rows")
        node_rows_a = batch.get("node_row_i", shared_rows)
        node_rows_b = batch.get("node_row_j", shared_rows)
        state_a = self.encode_node_state(
            batch["emb_a"],
            batch["len_a"],
            batch["x_a"],
            batch["ground_a"],
            batch.get("ground_id_a"),
            node_rows_a,
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
            (node_rows_b.index_select(0, non_self) if node_rows_b is not None else None),
        )
        if non_self.numel() == batch_size:
            return state_a, state_b_non_self, is_self
        return state_a, self._merge_node_states(state_a, state_b_non_self, non_self), is_self

    # ------------------------------------------------------------------ pair context

    def _stitch_and_encode(
        self,
        state_a: E2ENodeState,
        state_b: E2ENodeState,
        is_self: torch.Tensor,
        *,
        perturbation: ScaffoldInputPerturbation | None = None,
        precomputed_graph: ImaginedGraph | None = None,
    ) -> tuple[ImaginedGraph | None, GraphEmbedding | None, GraphEmbedding | None]:
        """Imagine the joint graph once, then encode both AB and BA directions.

        The graph is built exactly once per pair (design §4): `swapped()`
        shares ``adj``/``mask``/``aux`` by reference, only ``x`` is
        relabelled for the BA stream.

        Returns `(None, None, None)` when the generator imagines nothing for
        this pair (`NullGenerator.stitch` always; design §3.3) -- `self.encoder`
        is never called in that case, matching it not being constructed at
        all for a null-generator model (`__init__`).
        """
        graph: ImaginedGraph | None
        if precomputed_graph is not None:
            if perturbation is not None:
                raise ValueError("precomputed_graph cannot be combined with a perturbation")
            if self.encoder is None:
                raise ValueError("precomputed_graph requires a constructed graph encoder")
            graph = precomputed_graph
        else:
            graph = self.generator.stitch(
                self._generator_state(state_a),
                self._generator_state(state_b),
                is_self,
                perturbation=perturbation,
            )
        if graph is None:
            return None, None, None
        assert self.encoder is not None, (
            "a generator that emits a real ImaginedGraph must be paired with a constructed encoder"
        )
        embedding_ab = self.encoder(graph)
        embedding_ba = self.encoder(graph.swapped())
        return graph, embedding_ab, embedding_ba

    def _build_pair_context_and_graph(
        self,
        state_a: E2ENodeState,
        state_b: E2ENodeState,
        is_self: torch.Tensor,
        *,
        need_topo: bool,
        scaffold_input_perturbation: ScaffoldInputPerturbation | None = None,
        precomputed_graph: ImaginedGraph | None = None,
    ) -> tuple[E2EPairContext, ImaginedGraph | None, GraphEmbedding | None]:
        """Shared implementation behind `build_pair_context_from_states`.

        `build_pair_context_from_states` keeps its existing narrow
        ``E2EPairContext``-only return type for its other callers
        (`score_universe.py`, `src/experiments/probes.py`, tests) and simply
        discards the extra two values. `forward` needs the actual
        `ImaginedGraph` / AB `GraphEmbedding` objects -- not just their
        derived tensors -- to route auxiliary-loss computation through
        `generator.auxiliary_losses` / `encoder.auxiliary_losses` (design §6)
        against the *same* stitch+encode pass that produced the scored
        logits, rather than paying for a second one.
        """
        batch_size = state_a.encoded.size(0)
        if is_self.shape != (batch_size,):
            raise ValueError(f"is_self shape must be {(batch_size,)}, got {tuple(is_self.shape)}")
        if precomputed_graph is not None and scaffold_input_perturbation is not None:
            raise ValueError("precomputed_graph cannot be combined with a perturbation")
        if precomputed_graph is not None and self.encoder is None:
            raise ValueError("precomputed_graph requires a constructed graph encoder")
        # A generator with no constructed encoder (`self.encoder is None`,
        # `generator.name: "null"`) never has topology to build regardless of
        # what the caller requested -- clamped here, the single place every
        # `need_topo` path (`forward`, `build_pair_context*`) routes through,
        # rather than duplicated at each call site.
        need_topo = need_topo and self.encoder is not None
        topo_ab: torch.Tensor | None = None
        topo_ba: torch.Tensor | None = None
        pooled_ab: torch.Tensor | None = None
        pooled_ba: torch.Tensor | None = None
        plan: torch.Tensor | None = None
        log_plan: torch.Tensor | None = None
        graph: ImaginedGraph | None = None
        embedding_ab: GraphEmbedding | None = None
        if need_topo:
            graph, embedding_ab, embedding_ba = self._stitch_and_encode(
                state_a,
                state_b,
                is_self,
                perturbation=scaffold_input_perturbation,
                precomputed_graph=precomputed_graph,
            )
            if graph is not None:
                assert embedding_ab is not None and embedding_ba is not None
                plan = graph.aux["plan"]
                log_plan = graph.aux["log_plan"]
                topo_ab = embedding_ab.tokens
                topo_ba = embedding_ba.tokens
                pooled_ab = embedding_ab.pooled
                pooled_ba = embedding_ba.pooled
        context = E2EPairContext(
            encoded_a=state_a.encoded,
            encoded_b=state_b.encoded,
            len_a=state_a.length,
            len_b=state_b.length,
            topo_ab=topo_ab,
            topo_ba=topo_ba,
            plan=plan,
            log_plan=log_plan,
            pooled_ab=pooled_ab,
            pooled_ba=pooled_ba,
        )
        return context, graph, embedding_ab

    def build_pair_context_from_states(
        self,
        state_a: E2ENodeState,
        state_b: E2ENodeState,
        is_self: torch.Tensor,
        *,
        need_topo: bool = True,
        scaffold_input_perturbation: ScaffoldInputPerturbation | None = None,
        precomputed_graph: ImaginedGraph | None = None,
    ) -> E2EPairContext:
        """Build the shared pair context once from cacheable endpoint states."""
        context, _, _ = self._build_pair_context_and_graph(
            state_a,
            state_b,
            is_self,
            need_topo=need_topo,
            scaffold_input_perturbation=scaffold_input_perturbation,
            precomputed_graph=precomputed_graph,
        )
        return context

    def build_pair_context(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        need_topo: bool = True,
    ) -> E2EPairContext:
        """Encode endpoints and build one reusable pair context."""
        state_a, state_b, is_self = self._pair_node_states(batch)
        return self.build_pair_context_from_states(state_a, state_b, is_self, need_topo=need_topo)

    @torch.no_grad()
    def probe_states(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Export encoder token states for a fixed probe batch (read-only diagnostic).

        Runs only the generator + encoder pathway used to build the
        ``topo_ab`` conditioning tensor the trunk consumes -- no classifier
        trunk, no head, no gradient. Callers typically pass a batch of
        self-pairs (``x_a == x_b``, spec Sec 13.9) so each row's tokens
        describe one probe node's own generated ego-net.

        Args:
            batch: Pair batch with endpoint features plus required
                ``ground_a``/``ground_b`` tensors. Grounding ids are optional.

        Returns:
            Shape ``(B, V, d_model)`` AB-direction encoder token states.

        Raises:
            RuntimeError: For a null-generator model (`self.encoder is None`):
                there is no encoder token state to export.
        """
        if self.encoder is None:
            raise RuntimeError(
                "probe_states has no encoder token state to export: this model's "
                "generator (generator.name: 'null') never builds one"
            )
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

        Delegates AB/BA symmetrization to `self.classifier.forward` (design
        §4): that is where the one-trunk-batch invariant now lives, proven
        numerically identical to the pre-refactor `EgoStitchE2E.score_pair_context`
        by `tests/model/test_classifier_component.py`. ``context.encoded_a``/
        ``encoded_b`` are already the classifier's own `encode_tokens` output
        (`encode_node_state`), so they are handed to `PairInputs` as-is --
        `self.classifier.forward` must not, and does not, re-encode them.

        A null-generator model (`self.encoder is None`) never has topology to
        gate: `masks.topo` is clamped off regardless of its actual values, so
        `cond` stays `None` and the classifier runs exactly as unconditioned
        -- this is not a caller error, so it must not raise, unlike a
        genuinely topo-conditioned model whose caller requested topology from
        a `context` that was never built with it (design §12 P3 acceptance
        criterion 5).

        Args:
            context: See above.
            masks: See above.
            edge_mask: See above.
        """
        batch_size = context.encoded_a.size(0)
        device = context.encoded_a.device
        if masks is None:
            masks = masks_for_null(NULL_NONE, batch_size, device)
        distributed_training = self.training and dist.is_available() and dist.is_initialized()
        need_topo = self.encoder is not None and (bool(masks.topo.any()) or distributed_training)
        if need_topo and (context.topo_ab is None or context.topo_ba is None):
            raise ValueError("pair context does not contain topology tokens")
        cond: PairConditioning | None = None
        if need_topo:
            assert context.topo_ab is not None and context.topo_ba is not None
            # Prefer the encoder's own pooled summary when the context carries
            # it. The `tokens.mean(1)` rebuild is a *fallback* for contexts
            # built before `pooled_ab`/`pooled_ba` existed (or by a caller that
            # reassembled the NamedTuple positionally): for an encoder whose
            # `pooled` is a learned readout rather than a mean -- `grit_gmt`'s
            # PMA seeds -- the rebuild is a different quantity entirely, and
            # silently feeding it to the `film_logit`/`pooled_adapter` rungs
            # would test a mechanism nobody designed.
            cond = PairConditioning(
                ab=GraphEmbedding(
                    tokens=context.topo_ab,
                    pooled=(
                        context.pooled_ab
                        if context.pooled_ab is not None
                        else context.topo_ab.mean(dim=1)
                    ),
                ),
                ba=GraphEmbedding(
                    tokens=context.topo_ba,
                    pooled=(
                        context.pooled_ba
                        if context.pooled_ba is not None
                        else context.topo_ba.mean(dim=1)
                    ),
                ),
            )
        pair = PairInputs(
            tokens_a=context.encoded_a,
            tokens_b=context.encoded_b,
            len_a=context.len_a,
            len_b=context.len_b,
            edge_mask=edge_mask,
        )
        # `nn.Module.__call__` is typed to return `Any`; `PairClassifier.forward`
        # itself is annotated `-> torch.Tensor`, so this narrows back to what
        # the module actually returns rather than widening the contract.
        return cast(torch.Tensor, self.classifier(pair, cond, masks=masks))

    def forward(
        self, batch: Mapping[str, torch.Tensor], *, masks: HeadNullMasks | None = None
    ) -> dict[str, object]:
        """Score one pair batch under an optional head-null condition.

        Args:
            batch: Pair batch with ``emb_a``/``emb_b`` token streams,
                ``len_a``/``len_b`` lengths, and the ``x_a``/``x_b`` per-node
                features the generator's `encode_node` consumes.
            masks: Optional per-pair topo activity mask; ``None`` defaults to
                the topo pathway fully active (``NULL_NONE``).

        Returns:
            ``{"logits": (B,)}`` fused edge logits, plus ``{"graph",
            "embedding_ab"}`` -- this generator's own most recent
            `ImaginedGraph` and this encoder's own AB-direction
            `GraphEmbedding` -- whenever topology was actually computed
            (``rel_target`` present, a live topo mask, or distributed
            training). Callers that need `generator.auxiliary_losses` /
            `encoder.auxiliary_losses` (design §6, e.g.
            `train_egostitch.py`'s `_CompositeStep`) reuse these instead of
            paying for a second stitch+encode pass.
        """
        batch_size = batch["emb_a"].size(0)
        device = batch["emb_a"].device
        if masks is None:
            masks = masks_for_null(NULL_NONE, batch_size, device)
        has_relational_targets = "rel_target" in batch
        distributed_training = self.training and dist.is_available() and dist.is_initialized()
        need_topo = bool(masks.topo.any()) or has_relational_targets or distributed_training
        state_a, state_b, is_self = self._pair_node_states(batch)
        context, graph, embedding_ab = self._build_pair_context_and_graph(
            state_a,
            state_b,
            is_self,
            need_topo=need_topo,
        )
        output: dict[str, object] = {
            "logits": self.score_pair_context(
                context,
                masks=masks,
                edge_mask=batch.get("edge_mask"),
            )
        }
        if graph is not None and embedding_ab is not None:
            output["graph"] = graph
            output["embedding_ab"] = embedding_ab
        return output

    def decompose(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
        """Evaluate both logits without rebuilding node or pair state.

        Returns ``{"full", "f_logit"}`` -- the contract `score_universe.py`'s
        save path reads.
        """
        batch_size = context.encoded_a.size(0)
        device = context.encoded_a.device
        return {
            "full": self.score_pair_context(context),
            "f_logit": self.score_pair_context(
                context, masks=masks_for_null(NULL_ALL_HEAD, batch_size, device)
            ),
        }

    # ------------------------------------------------------------------ loss aggregation
    #
    # There is deliberately no `aggregate_losses` here. An earlier draft
    # combined `generator.auxiliary_losses` / `encoder.auxiliary_losses`
    # with `losses.stage1_total` / `.stage1_family_tensors` in one composite
    # method, but that duplicated policy that has to live in the trainer
    # anyway: `train_egostitch.py`'s `_CompositeStep` is the only caller with
    # `real_ssl_scale` (the Phase A/B warm-start ramp) in scope, and the
    # curriculum weighting it applies via `stage1_total`/`stage1_family_tensors`
    # is trainer policy, not model policy (three-component refactor design
    # §6/§12 P2 retargeting note) -- it must not move into this class. A
    # composite-owned version would also need its own `graph`/`embedding_ab`,
    # forcing a second stitch+encode pass unless the caller threaded through
    # the ones `forward` already computed, at which point the caller may as
    # well call `generator.auxiliary_losses` / `encoder.auxiliary_losses`
    # directly on `forward`'s own `"graph"`/`"embedding_ab"` output -- exactly
    # what `_CompositeStep.forward` does.
