"""``GraphEncoder`` base class and the relational auxiliary loss it owns.

Design 2026-08-02 §3.3, §6: a `GraphEncoder` consumes an `ImaginedGraph` and
emits a `GraphEmbedding`. It also owns `L_rel`, the train-only relational
head that predicts common-neighbour and Jaccard overlap from its own encoded
tokens (today `EgoStitchE2E.rel_head` + `relational_predictions`,
`e2e_model.py:340-349`, and `relational_loss`, `losses.py:147`). Owning the
loss here -- not in the composite -- is the point: swapping the encoder
swaps its auxiliary loss with it, because the loss reads only this
encoder's own `GraphEmbedding`, never the graph's generator-private `aux`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import nn

from src.model.egostitch.graph import GraphEmbedding, ImaginedGraph
from src.model.egostitch.losses import relational_loss


class GraphEncoder(nn.Module, ABC):
    """Consumes an `ImaginedGraph`, emits a `GraphEmbedding`.

    Concrete encoders must size every input-facing layer from the graph's
    runtime `feature_dim` / `num_relations` (`ImaginedGraph.feature_dim`,
    `.num_relations`), never from module-level constants -- that is the
    single change that makes this slot substitutable (design §1).

    Concrete encoders must never read `graph.aux`: it is generator-private,
    and reading it would recouple the encoder to one specific generator,
    defeating the point of the split (design §3.1).

    Subclasses get `L_rel` for free by calling
    `super().__init__(d_model=..., w_rel=...)` from their own `__init__`;
    passing `w_rel=0.0` disables the relational head entirely, mirroring
    today's `cfg.w_rel > 0.0` gate at `e2e_model.py:145`.

    Attributes:
        out_dim: Width `d` of both `GraphEmbedding.tokens` and `.pooled`.
    """

    out_dim: int

    def __init__(self, *, d_model: int, w_rel: float) -> None:
        """Build the shared relational head.

        Args:
            d_model: Width of the encoded tokens the relational head reads.
                Must equal the `d` a subclass's `forward` returns.
            w_rel: Nonnegative auxiliary-loss weight. `w_rel == 0.0` omits
                the relational head entirely (no extra parameters), exactly
                as `EgoStitchE2E.rel_head` is `None` when `cfg.w_rel == 0.0`.

        Raises:
            ValueError: If `w_rel` is negative.
        """
        super().__init__()
        if w_rel < 0.0:
            raise ValueError(f"w_rel must be non-negative, got {w_rel}")
        self.w_rel = w_rel
        self.rel_head: nn.Sequential | None = None
        if w_rel > 0.0:
            rel_hidden = max(1, d_model // 2)
            self.rel_head = nn.Sequential(
                nn.Linear(d_model, rel_hidden),
                nn.GELU(),
                nn.Linear(rel_hidden, 2),
            )

    @abstractmethod
    def forward(self, graph: ImaginedGraph) -> GraphEmbedding:
        """Encode one batch of imagined graphs.

        Must not read `graph.aux`.
        """
        raise NotImplementedError

    def relational_predictions(self, embedding: GraphEmbedding) -> torch.Tensor:
        """Predict the two train-only relational targets from encoded tokens.

        Mirrors `EgoStitchE2E.relational_predictions` (`e2e_model.py:340-349`):
        an unweighted mean over the token axis, evaluated in an fp32 island,
        fed to the relational head. Deliberately uses `embedding.tokens`, not
        `embedding.pooled` -- `pooled` is mask-weighted and new; nothing
        historically consumed it, so preserving today's numerics means
        keeping the plain mean here.

        Args:
            embedding: This encoder's own AB-direction output.

        Returns:
            Shape `(B, 2)` predictions.

        Raises:
            RuntimeError: If the relational head is absent (`w_rel == 0.0`).
        """
        if self.rel_head is None:
            raise RuntimeError("relational head is absent when w_rel == 0")
        with torch.autocast(device_type=embedding.tokens.device.type, enabled=False):
            pair_state = embedding.tokens.float().mean(dim=1)
            prediction: torch.Tensor = self.rel_head(pair_state)
            return prediction

    def auxiliary_losses(
        self, embedding: GraphEmbedding, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute `L_rel` (design §6).

        Args:
            embedding: This encoder's own AB-direction output for the batch.
            batch: Must carry `rel_target` (shape `(B, 2)`), `edge_mask`
                (shape `(B,)`), and `loss_world_size` (scalar) -- the same
                keys `EgoStitchE2E._training_relational_losses` reads today
                (`e2e_model.py:384-395`).

        Returns:
            `{"rel_loss": ...}`. A zero-valued tensor (gradient-connected to
            `embedding`, so it is safe to sum into a larger loss tree) when
            the relational head is disabled.
        """
        if self.rel_head is None:
            return {"rel_loss": embedding.tokens.sum() * 0.0}
        world_size = int(batch["loss_world_size"].item())
        return {
            "rel_loss": relational_loss(
                self.relational_predictions(embedding),
                batch["rel_target"],
                batch["edge_mask"],
                world_size=world_size,
            )
        }
