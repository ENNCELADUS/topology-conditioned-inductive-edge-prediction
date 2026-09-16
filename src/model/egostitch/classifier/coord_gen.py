"""Endpoint-only coordinate student with an adaptable interface and immutable teacher.

The encoder and cross-attention remain frozen. The generator, prompt interface
and output head train using the normalized task/KD mixture and coordinate loss;
anchor KD and structural supervision are assembled by the training stream.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from src.data.struct_coords import (
    COORD_DIM,
    COORD_NAMES,
    COORD_SPEC,
    ENDPOINT_DIM,
    FIELD_SLICES,
    get_coord_spec,
)
from src.model.egostitch.classifier.b0_v31 import unpack_pair_batch
from src.model.egostitch.classifier.layers import _build_padding_mask, masked_max, masked_mean
from src.model.egostitch.classifier.topo_prompt import COORDS_KEY, INTERVENTIONS, V3_1TopoPrompt
from src.model.egostitch.classifier.virtual_graph import VirtualGraphGenerator

DISTANCE_NAMES: tuple[str, ...] = ("dist_2", "dist_3", "dist_4plus", "dist_inf")
DISTANCE_INDEX: tuple[int, ...] = tuple(COORD_NAMES.index(name) for name in DISTANCE_NAMES)
#: Distance classes the generator predicts: the four one-hot categories plus the
#: all-zero case of a self-pair (``u == v`` carries no shortest-path category).
DISTANCE_CLASSES: int = len(DISTANCE_NAMES) + 1
SELF_DISTANCE_CLASS: int = len(DISTANCE_NAMES)
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


def distance_class_targets(coords: torch.Tensor, spec: str = COORD_SPEC) -> torch.Tensor:
    """Return the ``(B,)`` distance class of raw coordinates.

    The four one-hot categories map to classes ``0..3``; an all-zero row (a
    self-pair, which `StructCoordinateTable` gives no category) maps to
    `SELF_DISTANCE_CLASS` rather than being coerced into ``dist_2``.
    """
    layout = get_coord_spec(spec)
    one_hot = coords[:, list(layout.distance_indices)].float()
    return torch.where(
        one_hot.sum(dim=1) > 0.5,
        one_hot.argmax(dim=1),
        torch.full_like(one_hot[:, 0], layout.self_distance_class, dtype=torch.int64),
    )


@dataclass(frozen=True)
class VirtualGraphConfig:
    """Fixed size and attention choices of the coarse graph."""

    k: int = 256
    d_z: int = 128
    heads: int = 4

    def __post_init__(self) -> None:
        if self.k <= 0 or self.d_z <= 0 or self.heads <= 0 or self.d_z % self.heads:
            raise ValueError("virtual_graph sizes must be positive and d_z divisible by heads")


@dataclass(frozen=True)
class CoordGenConfig:
    """The ``model.config.coord_gen`` block.

    Attributes:
        reader_checkpoint: Path of the published Stage I ``v3_1_topo_prompt``
            checkpoint whose reader (and coordinate statistics) this
            model wraps. Provenance only once embedded.
        reader_checkpoint_sha256: SHA-256 of that file (provenance only, never verified).
        hidden: Hidden width of the two generator heads.
        layers: Hidden layers per head.
        dropout: Dropout inside the heads.
        w_coord: Weight of the coordinate-supervision term.
        kd_alpha: Pointwise KD fraction of the normalized classification mixture.
        w_anchor: Structural-stream anchor KL weight.
        anchor_temperature: Candidate-softmax temperature.
        w_kd_rep: Optional representation cosine KD weight.
        endpoint_dropout: Input dropout of the endpoint head.
        endpoint_weight_decay: Endpoint optimizer weight decay.
        endpoint_hidden: Endpoint hidden width.
        trainable: Generator-only or generator plus interface and output head.
        coord_spec: Coordinate specification the reader was trained on.
    """

    reader_checkpoint: str = ""
    reader_checkpoint_sha256: str | None = None
    hidden: int = 512
    layers: int = 2
    dropout: float = 0.1
    w_coord: float = 1.0
    kd_alpha: float = 0.5
    w_anchor: float = 1.0
    anchor_temperature: float = 1.0
    w_kd_rep: float = 0.0
    endpoint_dropout: float = 0.3
    endpoint_weight_decay: float = 0.1
    endpoint_hidden: int = 256
    trainable: str = "interface_head"
    coord_spec: str = COORD_SPEC
    generator: str = "mlp"
    virtual_graph: VirtualGraphConfig = field(default_factory=VirtualGraphConfig)

    def __post_init__(self) -> None:
        """Validate ranges.

        Raises:
            ValueError: On a non-positive size, an out-of-range dropout, a
                negative loss weight, or an unsupported spec.
        """
        if self.hidden <= 0 or self.layers <= 0:
            raise ValueError("coord_gen.hidden and coord_gen.layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("coord_gen.dropout must lie in [0, 1)")
        if not 0 <= self.kd_alpha <= 1:
            raise ValueError("coord_gen.kd_alpha must lie in [0, 1]")
        if self.trainable not in ("generator", "interface_head"):
            raise ValueError("coord_gen.trainable must be generator or interface_head")
        if (
            not 0 <= self.endpoint_dropout < 1
            or self.endpoint_hidden <= 0
            or self.endpoint_weight_decay < 0
        ):
            raise ValueError("invalid endpoint regularisation")
        if self.anchor_temperature <= 0:
            raise ValueError("anchor_temperature must be positive")
        weights = (self.w_coord, self.w_anchor, self.w_kd_rep)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("coord_gen loss weights must be non-negative")
        get_coord_spec(self.coord_spec)
        if self.generator not in ("mlp", "virtual_graph"):
            raise ValueError("coord_gen.generator must be mlp or virtual_graph")
        if self.generator == "virtual_graph" and self.coord_spec != "v3":
            raise ValueError("virtual_graph requires coord_spec v3")

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
        graph = raw.get("virtual_graph", {})
        if not isinstance(graph, Mapping):
            raise ValueError("coord_gen.virtual_graph must be a mapping")
        unknown_graph = set(graph) - set(VirtualGraphConfig.__dataclass_fields__)
        if unknown_graph:
            raise ValueError(f"unknown virtual_graph keys: {sorted(unknown_graph)}")
        graph_cfg = VirtualGraphConfig(
            k=int(graph.get("k", 256)),
            d_z=int(graph.get("d_z", 128)),
            heads=int(graph.get("heads", 4)),
        )
        sha = raw.get("reader_checkpoint_sha256")
        return cls(
            reader_checkpoint=str(raw.get("reader_checkpoint", "")),
            reader_checkpoint_sha256=None if sha is None else str(sha),
            hidden=int(cast(int, raw.get("hidden", 512))),
            layers=int(cast(int, raw.get("layers", 2))),
            dropout=float(cast(float, raw.get("dropout", 0.1))),
            w_coord=float(cast(float, raw.get("w_coord", 1.0))),
            kd_alpha=float(cast(float, raw.get("kd_alpha", 0.5))),
            w_anchor=float(cast(float, raw.get("w_anchor", 1.0))),
            anchor_temperature=float(cast(float, raw.get("anchor_temperature", 1.0))),
            w_kd_rep=float(cast(float, raw.get("w_kd_rep", 0.0))),
            endpoint_dropout=float(cast(float, raw.get("endpoint_dropout", 0.3))),
            endpoint_weight_decay=float(cast(float, raw.get("endpoint_weight_decay", 0.1))),
            endpoint_hidden=int(cast(int, raw.get("endpoint_hidden", 256))),
            trainable=str(raw.get("trainable", "interface_head")),
            coord_spec=str(raw.get("coord_spec", COORD_SPEC)),
            generator=str(raw.get("generator", "mlp")),
            virtual_graph=graph_cfg,
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
    relation coordinates, five shortest-path class logits (the four categories
    plus the self-pair "no category" case) and the five context coordinates.
    Every output is float32, whatever autocast the heads ran under.
    """

    def __init__(self, d_model: int, cfg: CoordGenConfig) -> None:
        """Build the heads.

        Args:
            d_model: Encoder width.
            cfg: The ``coord_gen`` block.
        """
        super().__init__()
        self.cfg = cfg
        self.spec = get_coord_spec(cfg.coord_spec)
        self.n_rel = self.spec.relation_dim - len(self.spec.distance_indices)
        self.distance_classes = self.spec.self_distance_class + 1
        pooled = 2 * d_model
        self.endpoint_head = nn.Sequential(
            nn.Dropout(cfg.endpoint_dropout),
            _mlp(2 * pooled, cfg.endpoint_hidden, cfg.layers, cfg.dropout, self.spec.endpoint_dim),
        )
        self.pair_head = _mlp(
            3 * pooled,
            cfg.hidden,
            cfg.layers,
            cfg.dropout,
            self.n_rel + self.distance_classes + self.spec.context_dim,
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
            ``(B, 7)`` standardised, ``distance_logits`` ``(B, 5)`` and
            ``context`` ``(B, 5)`` standardised, all float32.
        """
        pair = self.pair_head(torch.cat([p_u + p_v, p_u * p_v, (p_u - p_v).abs()], dim=-1))
        pair = pair.float()
        n_rel = self.n_rel
        return {
            "endpoint_u": self.endpoint_head(torch.cat([p_u, p_v], dim=-1)).float(),
            "endpoint_v": self.endpoint_head(torch.cat([p_v, p_u], dim=-1)).float(),
            "relation": pair[:, :n_rel],
            "distance_logits": pair[:, n_rel : n_rel + self.distance_classes],
            "context": pair[:, n_rel + self.distance_classes :],
        }


class V3_1CoordGen(nn.Module):
    """A Stage I reader driven by predicted coordinates and an adaptable interface.

    ``generator`` is registered before ``reader`` so ``next(model.parameters())``
    is a trainable parameter. Every forward predicts the coordinates; when the
    batch carries ``struct_coords`` (training and validation rows) the true
    coordinates supervise the prediction and drive the teacher logits, and with
    ``label`` the composite loss is returned. Scoring passes neither.
    """

    name: str = "v3_1_coord_gen"
    coordinate_scale: torch.Tensor

    def __init__(self, *, reader: Mapping[str, object], coord_gen: Mapping[str, object]) -> None:
        """Build a reader and teacher; initialize_teacher snapshots loaded Stage I weights.

        Args:
            reader: The Stage I checkpoint's ``model_config``
                (``{"base": ..., "topo_prompt": ...}``).
            coord_gen: The ``model.config.coord_gen`` block.
        """
        super().__init__()
        self.cfg = CoordGenConfig.from_mapping(coord_gen)
        self.spec = get_coord_spec(self.cfg.coord_spec)
        self.reader_config: dict[str, object] = dict(reader)
        reader_model = V3_1TopoPrompt(
            base=cast(Mapping[str, object], self.reader_config["base"]),
            topo_prompt=cast(Mapping[str, object], self.reader_config["topo_prompt"]),
        )
        if reader_model.cfg.coord_spec != self.cfg.coord_spec:
            raise ValueError("coord_gen and reader coord_spec must match")
        self.d_model = int(reader_model.d_model)
        self.input_dim = int(reader_model.input_dim)
        self.kd_rep_head = None
        self.kd_struct_head = None
        self.topo_gen = None
        self.generator: CoordinateGenerator | VirtualGraphGenerator
        if self.cfg.generator == "virtual_graph":
            graph_cfg = self.cfg.virtual_graph
            self.generator = VirtualGraphGenerator(
                self.d_model,
                k=graph_cfg.k,
                d_z=graph_cfg.d_z,
                heads=graph_cfg.heads,
            )
        else:
            self.generator = CoordinateGenerator(self.d_model, self.cfg)
        # Historical v1 checkpoints have no loss-scale state. Persist it once
        # training installs the nonself statistics; inference does not need it.
        self.register_buffer(
            "coordinate_scale", torch.ones(len(self.spec.continuous_indices)), persistent=False
        )
        self.register_load_state_dict_pre_hook(self._restore_coordinate_scale)  # type: ignore[no-untyped-call]
        self.reader = reader_model
        for param in self.reader.parameters():
            param.requires_grad_(False)
        self.reader.eval()
        self.teacher = deepcopy(self.reader).requires_grad_(False).eval()
        if self.cfg.trainable == "interface_head":
            self.reader.generator.requires_grad_(True)
            self.reader.base.output_head.requires_grad_(True)

    def _restore_coordinate_scale(
        self, module: nn.Module, state: Mapping[str, torch.Tensor], prefix: str, *args: object
    ) -> None:
        if prefix + "coordinate_scale" in state:
            self._non_persistent_buffers_set.discard("coordinate_scale")

    def install_coordinate_scale(self, coords: torch.Tensor) -> None:
        """Measure continuous residuals in nonself training standard deviations."""
        nonself = (
            distance_class_targets(coords, self.cfg.coord_spec) != self.spec.self_distance_class
        )
        if not nonself.any():
            raise ValueError("coordinate scale requires nonself training rows")
        stats = self.reader.generator
        cont = list(self.spec.continuous_indices)
        values = coords[nonself][:, cont].float().to(stats.coord_std.device)
        scale = values.std(dim=0, correction=0) / stats.coord_std[cont].float()
        if not torch.isfinite(scale).all():
            raise ValueError("non-finite nonself coordinate scale")
        self.coordinate_scale.copy_(scale.clamp_min(1e-6))
        self._non_persistent_buffers_set.discard("coordinate_scale")

    def initialize_teacher(self) -> None:
        """Snapshot the loaded Stage I reader once, before training starts."""
        self.teacher = deepcopy(self.reader).requires_grad_(False).eval()
        self.teacher.intervention = "none"
        if isinstance(self.generator, VirtualGraphGenerator):
            self.generator.coord_mean.copy_(self.reader.generator.coord_mean)
            self.generator.coord_std.copy_(self.reader.generator.coord_std)

    def optimizer_parameter_groups(
        self,
        generator_lr: float,
        interface_lr: float,
        weight_decay: float,
    ) -> list[dict[str, object]]:
        """Separate endpoint regularisation while sharing the generator schedule."""
        endpoint = (
            list(self.generator.endpoint_head.parameters())
            if isinstance(self.generator, CoordinateGenerator)
            else []
        )
        endpoint_ids = {id(p) for p in endpoint}
        groups: list[dict[str, object]] = [
            {
                "name": "endpoint",
                "params": endpoint,
                "lr": generator_lr,
                "max_lr": generator_lr,
                "weight_decay": self.cfg.endpoint_weight_decay,
            },
            {
                "name": "generator",
                "params": [p for p in self.generator.parameters() if id(p) not in endpoint_ids],
                "lr": generator_lr,
                "max_lr": generator_lr,
                "weight_decay": weight_decay,
            },
        ]
        groups = [group for group in groups if group["params"]]
        interface = [p for p in self.reader.parameters() if p.requires_grad]
        if interface:
            groups.append(
                {
                    "name": "interface",
                    "params": interface,
                    "lr": interface_lr,
                    "max_lr": interface_lr,
                    "weight_decay": weight_decay,
                }
            )
        return groups

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
        if value not in INTERVENTIONS:
            raise ValueError(f"unsupported coordinate intervention {value!r}")
        if value == "mean_context" and not self.spec.context_dim:
            raise ValueError("mean_context is unavailable without a context field")
        self.reader.intervention = value

    def train(self, mode: bool = True) -> V3_1CoordGen:
        """Train the generator and interface while the frozen trunk and teacher stay in eval.

        Args:
            mode: Training mode for the generator.

        Returns:
            ``self``.
        """
        super().train(mode)
        self.reader.eval()
        if self.cfg.trainable == "interface_head":
            self.reader.generator.train(mode)
            self.reader.base.output_head.train(mode)
        self.teacher.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters the optimiser updates: generator and configured reader interface."""
        return [p for p in self.parameters() if p.requires_grad]

    def assemble(self, parts: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Assemble the head outputs into ``(B, COORD_DIM)`` standardised coordinates.

        The four shortest-path one-hots enter as the softmax probabilities of
        their classes standardised with the reader's statistics, so the reader
        sees soft one-hots on the scale it was trained on; the probability mass
        of the self-pair class leaves all four at zero, the true self-pair value.
        Assembled in float32 regardless of autocast.
        """
        stats = self.reader.generator
        batch = parts["endpoint_u"].size(0)
        z = torch.zeros(
            (batch, self.spec.coord_dim), dtype=torch.float32, device=parts["endpoint_u"].device
        )
        z[:, self.spec.field_slices["endpoint_u"]] = parts["endpoint_u"].float()
        z[:, self.spec.field_slices["endpoint_v"]] = parts["endpoint_v"].float()
        relation = self.spec.field_slices["relation"]
        rel_cont = [
            i for i in range(relation.start, relation.stop) if i not in self.spec.distance_indices
        ]
        z[:, rel_cont] = parts["relation"].float()
        if self.spec.context_dim:
            z[:, self.spec.field_slices["context"]] = parts["context"].float()
        dist = list(self.spec.distance_indices)
        probs = torch.softmax(parts["distance_logits"].float(), dim=-1)[
            :, : len(self.spec.distance_names)
        ]
        z[:, dist] = (probs - stats.coord_mean[dist].float()) / stats.coord_std[dist].float()
        return z

    def predict(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(z_hat, parts)`` for encoded pairs (``a`` as endpoint ``u``)."""
        if self.intervention != "none" and self.training:
            raise ValueError("coordinate interventions are scoring-time only; call eval() first")
        if isinstance(self.generator, VirtualGraphGenerator):
            parts = self.generator(encoded_a, encoded_b, lengths_a, lengths_b)
        else:
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
            cross-entropy of the shortest-path class (`distance_class_targets`,
            self-pairs as their own class).
        """
        z_star = self.reader.generator.standardize(coords.to(z_hat.device))
        cont = list(self.spec.continuous_indices)
        residual = (z_hat[:, cont] - z_star[:, cont]) / self.coordinate_scale
        continuous = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none").mean(
            dim=1
        )
        target = distance_class_targets(coords.to(z_hat.device), self.cfg.coord_spec)
        distance = F.cross_entropy(parts["distance_logits"].float(), target, reduction="none")
        nonself = target != self.spec.self_distance_class
        continuous = continuous * nonself
        distance = distance * nonself
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
            with ``struct_coords`` also ``teacher_logits`` (unless ``kd_alpha == 0``),
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
        if self.cfg.kd_alpha > 0.0 or self.cfg.w_kd_rep > 0.0:
            with torch.no_grad():
                z_star = self.teacher.generator.standardize(coords.to(z_hat.device))
                teacher = self.teacher.logits_from_standardized(
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
        total_row = (1 - self.cfg.kd_alpha) * bce_row + self.cfg.w_coord * coord_row
        if kd_row is not None:
            total_row = total_row + self.cfg.kd_alpha * kd_row
            q = torch.sigmoid(output["teacher_logits"].reshape(-1).float())
            entropy = F.binary_cross_entropy_with_logits(
                output["teacher_logits"].reshape(-1).float(), q, reduction="none"
            )
            # Diagnostics are unweighted row means so `_coordinate_fit_metrics`
            # can aggregate them by row count into batching-invariant dataset means.
            output["teacher_entropy"] = entropy.detach().mean()
            output["kd_kl"] = (kd_row - entropy).detach().mean()
        if self.cfg.w_kd_rep > 0:
            student_repr = self.reader.logits_from_standardized(
                encoded_a, encoded_b, lengths_a, lengths_b, z_hat, return_pair_repr=True
            )
            with torch.no_grad():
                teacher_repr = self.teacher.logits_from_standardized(
                    encoded_a,
                    encoded_b,
                    lengths_a,
                    lengths_b,
                    self.teacher.generator.standardize(coords.to(z_hat.device)),
                    return_pair_repr=True,
                )
            rep_row = 1 - F.cosine_similarity(student_repr.float(), teacher_repr.float(), dim=-1)
            total_row = total_row + self.cfg.w_kd_rep * rep_row
            output["kd_rep_loss"] = rep_row.detach().mean()
        if isinstance(self.generator, VirtualGraphGenerator) and "attachment_target_u" in merged:
            attachment_rows = []
            for side in ("u", "v"):
                prediction = parts[f"attachment_counts_{side}"]
                target = merged[f"attachment_target_{side}"].to(prediction.device)
                if (target < 0).any():
                    raise ValueError(
                        "attachment supervision requires feature-present training nodes"
                    )
                attachment_rows.append(self.generator.attachment_loss_rows(prediction, target))
            attachment_row = (attachment_rows[0] + attachment_rows[1]) * 0.5
            # Node-prior supervision has equal endpoint/row weight, independent of label weights.
            output["attachment_loss"] = attachment_row.mean()
            for name, positive in (("nonzero", True), ("zero", False)):
                errors = []
                for side in ("u", "v"):
                    prediction = parts[f"attachment_counts_{side}"]
                    target = merged[f"attachment_target_{side}"].to(prediction.device)
                    mask = target > 0 if positive else target == 0
                    error = (prediction.log1p() - target.log1p()).abs()
                    errors.append((error * mask).sum(1) / mask.sum(1).clamp_min(1))
                output[f"attachment_{name}_error"] = ((errors[0] + errors[1]) * 0.5).detach().mean()
        output["loss"] = (weights * total_row).sum() / weight_sum
        if "attachment_loss" in output:
            output["loss"] = output["loss"] + output["attachment_loss"]
        output["loss_weight_sum"] = weight_sum
        output["task_loss"] = ((weights * bce_row).sum() / weight_sum).detach()
        return output


__all__ = [
    "CONTINUOUS_INDEX",
    "DISTANCE_CLASSES",
    "DISTANCE_INDEX",
    "DISTANCE_NAMES",
    "SELF_DISTANCE_CLASS",
    "distance_class_targets",
    "FIELD_CONTINUOUS_INDEX",
    "RELATION_CONTINUOUS_INDEX",
    "CoordGenConfig",
    "CoordinateGenerator",
    "V3_1CoordGen",
]
