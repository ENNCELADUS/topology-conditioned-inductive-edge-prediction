"""Stage II of the topology-representation-transfer pipeline: predicted coordinates.

`V3_1CoordGen` wraps a published Stage I reader (`V3_1TopoPrompt`: encoder,
trunk, prompt interface, head and coordinate statistics, all frozen) and trains
only a coordinate generator ``g`` that predicts the queried pair's *standardised*
structural coordinates (`src.data.struct_coords`, spec ``v1``) from the frozen
encoder's endpoint token states. The reader then reads the prediction through
the same prompt interface it learned on true structure, so the deployable model
is a function of ``(x_u, x_v)`` alone and is scored without any truth graph.

The generator is swap-equivariant by construction: the endpoint head is applied
as ``e(p_u, p_v)`` and ``e(p_v, p_u)``, and the pair head reads only symmetric
combinations of the two pooled endpoints, so swapping the pair swaps the two
endpoint fields and fixes the relation and context fields -- exactly the
symmetry of the true coordinates.

Training rows carry the true coordinates (measured on the training graph with
the query edge removed) as ``batch["struct_coords"]``. The loss is a row-weighted
mean (the base's positive weight, so the trainer's DDP scaling applies) of
``w_task * BCE + w_kd * KD + w_coord * coordinate``: Huber on the 30 continuous
standardised coordinates plus cross-entropy over the four shortest-path classes,
and a soft-target logit KD towards the same frozen reader fed the true
coordinates (the Stage I teacher copy, which shares every weight with the
student's reader in this stage).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from src.data.struct_coords import (
    CONTEXT_DIM,
    COORD_DIM,
    COORD_NAMES,
    COORD_SPEC,
    ENDPOINT_DIM,
    FIELD_SLICES,
)
from src.model.egostitch.classifier.b0_v31 import unpack_pair_batch
from src.model.egostitch.classifier.layers import _build_padding_mask, masked_max, masked_mean
from src.model.egostitch.classifier.topo_prompt import COORDS_KEY, V3_1TopoPrompt

DISTANCE_NAMES: tuple[str, ...] = ("dist_2", "dist_3", "dist_4plus", "dist_inf")
DISTANCE_INDEX: tuple[int, ...] = tuple(COORD_NAMES.index(name) for name in DISTANCE_NAMES)
CONTINUOUS_INDEX: tuple[int, ...] = tuple(
    index for index in range(COORD_DIM) if index not in DISTANCE_INDEX
)
RELATION_CONTINUOUS_INDEX: tuple[int, ...] = tuple(
    index
    for index in range(FIELD_SLICES["relation"].start, FIELD_SLICES["relation"].stop)
    if index not in DISTANCE_INDEX
)
#: Continuous coordinate indices per reported field (both endpoint fields pooled).
FIELD_CONTINUOUS_INDEX: dict[str, tuple[int, ...]] = {
    "endpoint": tuple(range(0, 2 * ENDPOINT_DIM)),
    "relation": RELATION_CONTINUOUS_INDEX,
    "context": tuple(range(FIELD_SLICES["context"].start, FIELD_SLICES["context"].stop)),
}


@dataclass(frozen=True)
class CoordGenConfig:
    """The ``model.config.coord_gen`` block.

    Attributes:
        reader_checkpoint: Path of the published Stage I ``v3_1_topo_prompt``
            checkpoint whose frozen reader (and coordinate statistics) this
            model wraps. Provenance only once embedded.
        reader_checkpoint_sha256: SHA-256 of that file (provenance only, never verified).
        hidden: Hidden width of the two generator heads.
        layers: Hidden layers per head.
        dropout: Dropout inside the heads.
        w_coord: Weight of the coordinate-supervision term.
        w_task: Weight of the student task BCE.
        w_kd: Weight of the logit KD towards the reader on true coordinates.
        coord_spec: Coordinate specification the reader was trained on.
    """

    reader_checkpoint: str = ""
    reader_checkpoint_sha256: str | None = None
    hidden: int = 512
    layers: int = 2
    dropout: float = 0.1
    w_coord: float = 1.0
    w_task: float = 1.0
    w_kd: float = 0.1
    coord_spec: str = COORD_SPEC

    def __post_init__(self) -> None:
        """Validate ranges.

        Raises:
            ValueError: On a non-positive size, an out-of-range dropout, a
                negative or all-zero loss weight, or an unsupported spec.
        """
        if self.hidden <= 0 or self.layers <= 0:
            raise ValueError("coord_gen.hidden and coord_gen.layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("coord_gen.dropout must lie in [0, 1)")
        weights = (self.w_coord, self.w_task, self.w_kd)
        if any(weight < 0.0 for weight in weights) or not any(weight > 0.0 for weight in weights):
            raise ValueError(
                "coord_gen loss weights must be non-negative with at least one positive"
            )
        if self.coord_spec != COORD_SPEC:
            raise ValueError(
                f"coord_gen.coord_spec {self.coord_spec!r} is not the supported {COORD_SPEC!r}"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> CoordGenConfig:
        """Parse the block, rejecting unknown keys.

        Args:
            raw: The mapping under ``model.config.coord_gen``.

        Returns:
            The parsed config.

        Raises:
            ValueError: On unknown keys.
        """
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown coord_gen keys: {unknown}")
        sha = raw.get("reader_checkpoint_sha256")
        return cls(
            reader_checkpoint=str(raw.get("reader_checkpoint", "")),
            reader_checkpoint_sha256=None if sha is None else str(sha),
            hidden=int(cast(int, raw.get("hidden", 512))),
            layers=int(cast(int, raw.get("layers", 2))),
            dropout=float(cast(float, raw.get("dropout", 0.1))),
            w_coord=float(cast(float, raw.get("w_coord", 1.0))),
            w_task=float(cast(float, raw.get("w_task", 1.0))),
            w_kd=float(cast(float, raw.get("w_kd", 0.1))),
            coord_spec=str(raw.get("coord_spec", COORD_SPEC)),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the block as a plain mapping (checkpoint-embeddable)."""
        return cast(dict[str, object], asdict(self))


def _mlp(in_dim: int, hidden: int, layers: int, dropout: float, out_dim: int) -> nn.Sequential:
    """``layers`` GELU hidden blocks with LayerNorm and dropout, then a linear output."""
    blocks: list[nn.Module] = []
    width = in_dim
    for _ in range(layers):
        blocks += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
        width = hidden
    blocks.append(nn.Linear(width, out_dim))
    return nn.Sequential(*blocks)


class CoordinateGenerator(nn.Module):
    """``g``: pooled endpoint states -> standardised coordinates, swap-equivariant.

    Each endpoint is pooled to ``[masked mean | masked max]`` of its encoder
    tokens (``2 * d_model``). The endpoint head reads ``[self | partner]`` and is
    applied once per endpoint; the pair head reads
    ``[p_u + p_v | p_u * p_v | |p_u - p_v|]`` and emits the seven continuous
    relation coordinates, four shortest-path class logits and the five context
    coordinates.
    """

    def __init__(self, d_model: int, cfg: CoordGenConfig) -> None:
        """Build the heads.

        Args:
            d_model: Encoder width.
            cfg: The ``coord_gen`` block.
        """
        super().__init__()
        self.cfg = cfg
        pooled = 2 * d_model
        self.endpoint_head = _mlp(2 * pooled, cfg.hidden, cfg.layers, cfg.dropout, ENDPOINT_DIM)
        self.pair_head = _mlp(
            3 * pooled,
            cfg.hidden,
            cfg.layers,
            cfg.dropout,
            len(RELATION_CONTINUOUS_INDEX) + len(DISTANCE_NAMES) + CONTEXT_DIM,
        )

    @staticmethod
    def pool(encoded: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        """Pool ``(B, L, d)`` token states to ``(B, 2d)`` over the valid positions."""
        pad = _build_padding_mask(lengths, encoded.size(1))
        keep = (
            torch.ones(encoded.shape[:2], dtype=torch.bool, device=encoded.device)
            if pad is None
            else ~pad
        )
        states = encoded.float()
        return torch.cat([masked_mean(states, keep), masked_max(states, keep)], dim=-1)

    def forward(self, p_u: torch.Tensor, p_v: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict the field pieces from the two pooled endpoints.

        Args:
            p_u: Pooled endpoint ``u`` ``(B, 2d)``.
            p_v: Pooled endpoint ``v`` ``(B, 2d)``.

        Returns:
            ``endpoint_u`` / ``endpoint_v`` ``(B, 9)`` standardised, ``relation``
            ``(B, 7)`` standardised, ``distance_logits`` ``(B, 4)`` and
            ``context`` ``(B, 5)`` standardised.
        """
        pair = self.pair_head(torch.cat([p_u + p_v, p_u * p_v, (p_u - p_v).abs()], dim=-1))
        n_rel = len(RELATION_CONTINUOUS_INDEX)
        n_dist = len(DISTANCE_NAMES)
        return {
            "endpoint_u": self.endpoint_head(torch.cat([p_u, p_v], dim=-1)),
            "endpoint_v": self.endpoint_head(torch.cat([p_v, p_u], dim=-1)),
            "relation": pair[:, :n_rel],
            "distance_logits": pair[:, n_rel : n_rel + n_dist],
            "context": pair[:, n_rel + n_dist :],
        }


class V3_1CoordGen(nn.Module):
    """A frozen Stage I reader driven by a trainable coordinate generator.

    ``generator`` is registered before ``reader`` so ``next(model.parameters())``
    is a trainable parameter. Every forward predicts the coordinates; when the
    batch carries ``struct_coords`` (training and validation rows) the true
    coordinates supervise the prediction and drive the teacher logits, and with
    ``label`` the composite loss is returned. Scoring passes neither.
    """

    name: str = "v3_1_coord_gen"

    def __init__(self, *, reader: Mapping[str, object], coord_gen: Mapping[str, object]) -> None:
        """Build the frozen reader from its checkpointed config and the generator on top.

        Args:
            reader: The Stage I checkpoint's ``model_config``
                (``{"base": ..., "topo_prompt": ...}``).
            coord_gen: The ``model.config.coord_gen`` block.
        """
        super().__init__()
        self.cfg = CoordGenConfig.from_mapping(coord_gen)
        self.reader_config: dict[str, object] = dict(reader)
        reader_model = V3_1TopoPrompt(
            base=cast(Mapping[str, object], self.reader_config["base"]),
            topo_prompt=cast(Mapping[str, object], self.reader_config["topo_prompt"]),
        )
        self.d_model = int(reader_model.d_model)
        self.input_dim = int(reader_model.input_dim)
        self.kd_rep_head = None
        self.kd_struct_head = None
        self.topo_gen = None
        self.generator = CoordinateGenerator(self.d_model, self.cfg)
        self.reader = reader_model
        for param in self.reader.parameters():
            param.requires_grad_(False)
        self.reader.eval()

    @property
    def encoder(self) -> nn.Module:
        """The reader's frozen per-node encoder (packed scoring caches its output)."""
        return self.reader.encoder

    @property
    def intervention(self) -> str:
        """The reader's scoring-time intervention (applied to the predicted coordinates)."""
        return self.reader.intervention

    @intervention.setter
    def intervention(self, value: str) -> None:
        self.reader.intervention = value

    def train(self, mode: bool = True) -> V3_1CoordGen:
        """Switch mode; the reader stays in eval mode regardless.

        Args:
            mode: Training mode for the generator.

        Returns:
            ``self``.
        """
        super().train(mode)
        self.reader.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters the optimiser updates: the generator only."""
        return list(self.generator.parameters())

    def assemble(self, parts: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Assemble the head outputs into ``(B, COORD_DIM)`` standardised coordinates.

        The four shortest-path classes enter as softmax probabilities standardised
        with the reader's statistics, so the reader sees soft one-hots on the
        scale it was trained on.
        """
        stats = self.reader.generator
        batch = parts["endpoint_u"].size(0)
        z = parts["endpoint_u"].new_zeros((batch, COORD_DIM))
        z[:, FIELD_SLICES["endpoint_u"]] = parts["endpoint_u"]
        z[:, FIELD_SLICES["endpoint_v"]] = parts["endpoint_v"]
        z[:, list(RELATION_CONTINUOUS_INDEX)] = parts["relation"]
        z[:, FIELD_SLICES["context"]] = parts["context"]
        dist = list(DISTANCE_INDEX)
        probs = torch.softmax(parts["distance_logits"].float(), dim=-1)
        z[:, dist] = (probs - stats.coord_mean[dist]) / stats.coord_std[dist]
        return z

    def predict(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(z_hat, parts)`` for encoded pairs (``a`` as endpoint ``u``)."""
        p_a = CoordinateGenerator.pool(encoded_a, lengths_a)
        p_b = CoordinateGenerator.pool(encoded_b, lengths_b)
        parts = self.generator(p_a, p_b)
        return self.assemble(parts), parts

    def logits_from_encoded(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> torch.Tensor:
        """Student logits from encoded token states (the packed scorer's path).

        Args:
            encoded_a: Encoder output for item A ``(B, L_a, d_model)``.
            encoded_b: Encoder output for item B ``(B, L_b, d_model)``.
            lengths_a: True sequence lengths for A.
            lengths_b: True sequence lengths for B.

        Returns:
            The pair logits under the predicted coordinates.
        """
        z_hat, _ = self.predict(encoded_a, encoded_b, lengths_a, lengths_b)
        return self.reader.logits_from_standardized(
            encoded_a, encoded_b, lengths_a, lengths_b, z_hat
        )

    def coordinate_loss_rows(
        self, parts: Mapping[str, torch.Tensor], z_hat: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-row coordinate supervision against raw true coordinates.

        Args:
            parts: The generator's head outputs.
            z_hat: The assembled standardised prediction ``(B, COORD_DIM)``.
            coords: Raw true coordinates ``(B, COORD_DIM)``.

        Returns:
            ``(total, continuous, distance)`` per-row losses: their sum, the mean
            Huber over the 30 continuous standardised coordinates, and the
            cross-entropy of the shortest-path class.
        """
        z_star = self.reader.generator.standardize(coords.to(z_hat.device))
        cont = list(CONTINUOUS_INDEX)
        continuous = F.smooth_l1_loss(z_hat[:, cont], z_star[:, cont], reduction="none").mean(dim=1)
        target = coords.to(z_hat.device)[:, list(DISTANCE_INDEX)].float().argmax(dim=1)
        distance = F.cross_entropy(parts["distance_logits"].float(), target, reduction="none")
        return continuous + distance, continuous, distance

    def forward(
        self,
        batch: dict[str, torch.Tensor] | None = None,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score pairs under predicted coordinates; supervise when truth rides along.

        Args:
            batch: Optional batch dictionary.
            **kwargs: Additional batch tensors merged into ``batch``.

        Returns:
            ``logits``, ``predicted_coords`` (standardised) and ``distance_logits``;
            with ``struct_coords`` also ``teacher_logits`` (unless ``w_kd == 0``),
            ``coord_loss``, ``coord_continuous_loss``, ``coord_distance_loss`` and
            ``kd_loss`` (detached means); with ``label`` too, the composite
            row-weighted ``loss``, its ``loss_weight_sum`` and the detached
            ``task_loss``. Without ``struct_coords`` no loss is returned even if a
            label is present, so a training step without attached targets fails
            closed on the missing key.
        """
        merged: dict[str, torch.Tensor] = {}
        if batch is not None:
            merged.update(batch)
        merged.update(kwargs)
        emb_a, emb_b, lengths_a, lengths_b = unpack_pair_batch(merged, self.input_dim)
        with torch.no_grad():
            encoded_a = self.reader.encoder(emb_a, lengths_a)
            encoded_b = self.reader.encoder(emb_b, lengths_b)
        z_hat, parts = self.predict(encoded_a, encoded_b, lengths_a, lengths_b)
        logits = self.reader.logits_from_standardized(
            encoded_a, encoded_b, lengths_a, lengths_b, z_hat
        )
        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "predicted_coords": z_hat,
            "distance_logits": parts["distance_logits"],
        }
        coords = merged.get(COORDS_KEY)
        if coords is None:
            # Evaluation and scoring forwards (the V_val universe pass, packed
            # scoring) carry labels but no coordinates: logits only, no loss, so a
            # training step without attached targets fails on the missing key.
            return output
        coord_row, continuous_row, distance_row = self.coordinate_loss_rows(parts, z_hat, coords)
        output["coord_loss"] = coord_row.detach().mean()
        output["coord_continuous_loss"] = continuous_row.detach().mean()
        output["coord_distance_loss"] = distance_row.detach().mean()
        flat = logits.reshape(-1).float()
        kd_row: torch.Tensor | None = None
        if self.cfg.w_kd > 0.0:
            with torch.no_grad():
                z_star = self.reader.generator.standardize(coords.to(z_hat.device))
                teacher = self.reader.logits_from_standardized(
                    encoded_a, encoded_b, lengths_a, lengths_b, z_star
                )
            output["teacher_logits"] = teacher
            kd_row = F.binary_cross_entropy_with_logits(
                flat, torch.sigmoid(teacher.reshape(-1).float()), reduction="none"
            )
            output["kd_loss"] = kd_row.detach().mean()
        if "label" not in merged:
            return output
        labels = merged["label"].reshape(-1).float()
        base = self.reader.base
        smoothing = float(base.label_smoothing)
        targets = labels * (1.0 - smoothing) + 0.5 * smoothing if smoothing > 0.0 else labels
        bce_row = F.binary_cross_entropy_with_logits(flat, targets, reduction="none")
        weights = 1.0 + (float(base.positive_weight) - 1.0) * labels
        weight_sum = weights.sum().detach()
        total_row = self.cfg.w_task * bce_row + self.cfg.w_coord * coord_row
        if kd_row is not None:
            total_row = total_row + self.cfg.w_kd * kd_row
        output["loss"] = (weights * total_row).sum() / weight_sum
        output["loss_weight_sum"] = weight_sum
        output["task_loss"] = ((weights * bce_row).sum() / weight_sum).detach()
        return output


__all__ = [
    "CONTINUOUS_INDEX",
    "DISTANCE_INDEX",
    "DISTANCE_NAMES",
    "FIELD_CONTINUOUS_INDEX",
    "RELATION_CONTINUOUS_INDEX",
    "CoordGenConfig",
    "CoordinateGenerator",
    "V3_1CoordGen",
]
