"""Fixed motif-token dictionary with equivariant endpoint-sequence routing."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from src.data.motif_dictionary import (
    FIELD_ORDER as ARTIFACT_FIELD_ORDER,
)
from src.data.motif_dictionary import (
    DictionaryArtifact,
    validate_dictionary,
)
from src.model.egostitch.classifier.b0_v31 import unpack_pair_batch
from src.model.egostitch.classifier.motif_prompt import (
    TEMPLATE_KEY,
    TEMPLATE_MASK_KEY,
    V3_1MotifPrompt,
)

ROUTE_TARGET_KEY = "motif_route_targets"
ROUTE_NONEMPTY_KEY = "motif_route_nonempty"
DICTIONARY_MODES = ("oracle", "sequence")
DICTIONARY_INTERVENTIONS = ("none", "gates_off", "mean", "shuffle_route")

# Slot order is [u, v, C1..C8, L1..L8, R1..R8].
_SLOT_SWAP = (1, 0, *range(2, 10), *range(18, 26), *range(10, 18))


@dataclass(frozen=True)
class MotifDictionaryConfig:
    """The ``model.config.motif_dictionary`` block."""

    mode: str
    artifact_path: str
    dictionary_size: int
    slot_checkpoint: str

    def __post_init__(self) -> None:
        """Validate enum and shape-driving fields."""
        if self.mode not in DICTIONARY_MODES:
            raise ValueError(f"motif_dictionary.mode must be one of {list(DICTIONARY_MODES)}")
        if self.dictionary_size <= 0:
            raise ValueError("motif_dictionary.dictionary_size must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> MotifDictionaryConfig:
        """Parse a checkpoint-embeddable mapping and reject unknown keys."""
        allowed = {"mode", "artifact_path", "dictionary_size", "slot_checkpoint"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown motif_dictionary keys: {unknown}")
        try:
            return cls(
                mode=str(raw["mode"]),
                artifact_path=str(raw.get("artifact_path", "")),
                dictionary_size=int(cast(int, raw["dictionary_size"])),
                slot_checkpoint=str(raw.get("slot_checkpoint", "")),
            )
        except KeyError as error:
            raise ValueError(f"motif_dictionary requires {error.args[0]}") from error


class MotifDictionaryRouter(nn.Module):
    """Shared MLP made endpoint-equivariant by explicit group averaging."""

    def __init__(self, dictionary_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(26 * 96, 256),
            nn.GELU(),
            nn.Linear(256, dictionary_size),
        )

    def forward(self, slots: torch.Tensor, swap_index: torch.Tensor) -> torch.Tensor:
        """Return endpoint-equivariant dictionary logits."""
        if tuple(slots.shape[1:]) != (26, 96):
            raise ValueError(f"router slots must have shape (B, 26, 96), got {tuple(slots.shape)}")
        direct = self.mlp(slots.flatten(1))
        swapped_slots = slots[:, _SLOT_SWAP]
        swapped = self.mlp(swapped_slots.flatten(1))[:, swap_index]
        return cast(torch.Tensor, (direct + swapped) * 0.5)


class V3_1MotifDictionary(V3_1MotifPrompt):
    """A motif prompt whose four fields are mixtures of a fixed token bank."""

    name: str = "v3_1_motif_dictionary"
    dictionary_weights: torch.Tensor
    dictionary_tokens: torch.Tensor
    dictionary_scales: torch.Tensor
    dictionary_temperature: torch.Tensor
    dictionary_mean_weights: torch.Tensor
    dictionary_swap_index: torch.Tensor
    dictionary_installed: torch.Tensor

    def __init__(
        self,
        *,
        base: Mapping[str, object],
        motif_prompt: Mapping[str, object],
        motif_dictionary: Mapping[str, object],
    ) -> None:
        self.dictionary_cfg = MotifDictionaryConfig.from_mapping(motif_dictionary)
        super().__init__(base=base, motif_prompt=motif_prompt)
        if self.dictionary_cfg.mode == "sequence" and self.generator is None:
            raise ValueError("sequence motif dictionary requires a stage-two motif_prompt")

        size = self.dictionary_cfg.dictionary_size
        self.router = MotifDictionaryRouter(size)
        self.register_buffer("dictionary_weights", torch.zeros(size, 96))
        self.register_buffer("dictionary_tokens", torch.zeros(size, 4, self.cfg.width))
        self.register_buffer("dictionary_scales", torch.ones(4))
        self.register_buffer("dictionary_temperature", torch.ones(()))
        self.register_buffer("dictionary_mean_weights", torch.full((size,), 1.0 / size))
        self.register_buffer("dictionary_swap_index", torch.arange(size, dtype=torch.long))
        self.register_buffer("dictionary_installed", torch.tensor(False))

        # The prefix/trunk/head form a fixed evaluator.  Sequence mode trains
        # only the repaired wave-3 slot reads and the new router; oracle mode
        # has no optimiser-owned parameters.
        self.requires_grad_(False)
        if self.dictionary_cfg.mode == "sequence":
            assert self.generator is not None
            for name, param in self.generator.named_parameters():
                if name.startswith(
                    (
                        "residue_proj.",
                        "attention.",
                        "bridge_queries",
                        "witness_queries",
                        "bridge_read.",
                        "witness_read.",
                        "endpoint_proj.",
                        "witness_mix.",
                    )
                ):
                    param.requires_grad_(True)
            self.router.requires_grad_(True)

    @property
    def mode(self) -> str:
        """Dictionary execution mode used by dispatch and scoring."""
        return self.dictionary_cfg.mode

    @property
    def dictionary_size(self) -> int:
        """Effective routing-bank width embedded in this model."""
        return self.dictionary_cfg.dictionary_size

    def train(self, mode: bool = True) -> V3_1MotifDictionary:
        """Keep every fixed producer/evaluator in eval during router training."""
        super().train(mode)
        self.reader.eval()
        self.count_head.eval()
        self.adapter.eval()
        self.prompt_layers.eval()
        self.base.eval()
        if self.generator is not None:
            self.generator.eval()
            if self.mode == "sequence":
                self.generator.residue_proj.train(mode)
                self.generator.attention.train(mode)
                self.generator.endpoint_proj.train(mode)
                self.generator.witness_mix.train(mode)
                if self.generator.bridge_read is not None:
                    self.generator.bridge_read.train(mode)
                if self.generator.witness_read is not None:
                    self.generator.witness_read.train(mode)
        self.router.train(mode and self.mode == "sequence")
        return self

    @torch.no_grad()
    def install_dictionary(self, artifact: DictionaryArtifact) -> None:
        """Copy one construction artifact into checkpoint-persistent buffers."""
        artifact = validate_dictionary(artifact)
        expected = self.dictionary_cfg.dictionary_size
        tensors = {
            "weights": torch.as_tensor(artifact.weights, dtype=torch.float32),
            "tokens": torch.as_tensor(artifact.tokens, dtype=torch.float32),
            "scales": torch.as_tensor(artifact.scales, dtype=torch.float32),
            "mean_weights": torch.as_tensor(artifact.mean_weights, dtype=torch.float32),
            "swap_index": torch.as_tensor(artifact.swap_index, dtype=torch.long),
        }
        shapes = {
            "weights": (expected, 96),
            "tokens": (expected, 4, self.cfg.width),
            "scales": (4,),
            "mean_weights": (expected,),
            "swap_index": (expected,),
        }
        for name, value in tensors.items():
            if tuple(value.shape) != shapes[name]:
                raise ValueError(
                    f"dictionary {name} must have shape {shapes[name]}, "
                    f"got {tuple(value.shape)}"
                )
            if name != "swap_index" and not torch.isfinite(value).all():
                raise ValueError(f"dictionary {name} must be finite")
        swap = tensors["swap_index"]
        if bool(((swap < 0) | (swap >= expected)).any()) or not torch.equal(
            swap[swap], torch.arange(expected, device=swap.device)
        ):
            raise ValueError("dictionary swap_index must be an in-range involution")
        temperature = float(artifact.temperature)
        if not temperature > 0.0 or not torch.isfinite(torch.tensor(temperature)):
            raise ValueError("dictionary temperature must be finite and positive")
        if bool((tensors["scales"] <= 0.0).any()):
            raise ValueError("dictionary scales must be positive")

        self.dictionary_weights.copy_(tensors["weights"].to(self.dictionary_weights))
        self.dictionary_tokens.copy_(tensors["tokens"].to(self.dictionary_tokens))
        self.dictionary_scales.copy_(tensors["scales"].to(self.dictionary_scales))
        self.dictionary_temperature.fill_(temperature)
        self.dictionary_mean_weights.copy_(
            tensors["mean_weights"].to(self.dictionary_mean_weights)
        )
        self.dictionary_swap_index.copy_(swap.to(self.dictionary_swap_index))
        self.dictionary_installed.fill_(True)

    def _require_dictionary(self) -> None:
        if not bool(self.dictionary_installed):
            raise RuntimeError("motif dictionary constants are not installed")

    @torch.no_grad()
    def routes_from_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Derive oracle q* from compiled templates and embedded constants."""
        self._require_dictionary()
        weights = weights.to(device=self.dictionary_tokens.device, dtype=torch.float32)
        with torch.autocast(device_type=weights.device.type, enabled=False):
            tokens = self.reader(weights)
            tokens["topo_cnt"] = self.count_head(weights)
            target = torch.stack(
                [tokens[name] for name in ARTIFACT_FIELD_ORDER], dim=1
            ).float()
            bank = self.dictionary_tokens.float()
            scales = self.dictionary_scales.float()
            distance = ((target[:, None] - bank[None]) / scales[None, None, :, None]).square()
            distance = distance.mean(dim=(-1, -2))
            return torch.softmax(
                -distance / self.dictionary_temperature.float(), dim=-1
            )

    def predict_routes(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> torch.Tensor:
        """Predict swap-equivariant routes from endpoint residue states."""
        return torch.softmax(
            self._route_logits(encoded_a, encoded_b, lengths_a, lengths_b), dim=-1
        )

    def _route_logits(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> torch.Tensor:
        """Return FP32 pre-softmax routes for stable KL arithmetic."""
        self._require_dictionary()
        if self.mode != "sequence" or self.generator is None:
            raise RuntimeError("predict_routes is available only in sequence mode")
        slots = self.generator._slot_states(  # noqa: SLF001
            encoded_a, encoded_b, lengths_a, lengths_b
        )
        with torch.autocast(device_type=slots.device.type, enabled=False):
            return cast(
                torch.Tensor,
                self.router(slots.float(), self.dictionary_swap_index).float(),
            )

    def logits_from_routes(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
        *,
        routes: torch.Tensor,
        return_pair_repr: bool = False,
    ) -> torch.Tensor:
        """Mix fixed tokens in FP32 and run the frozen four-field interface."""
        self._require_dictionary()
        if tuple(routes.shape) != (encoded_a.size(0), self.dictionary_cfg.dictionary_size):
            raise ValueError(
                "routes must have shape "
                f"({encoded_a.size(0)}, {self.dictionary_cfg.dictionary_size})"
            )
        if self.intervention not in DICTIONARY_INTERVENTIONS:
            raise ValueError(f"unknown motif dictionary intervention {self.intervention!r}")
        if self.intervention == "shuffle_route":
            raise ValueError(
                "shuffle_route is a whole-universe scorer substitution; pass explicit routes "
                "and leave the model intervention on 'none'"
            )
        if self.intervention != "none" and self.training:
            raise ValueError(
                "motif dictionary interventions are scoring-time only; call eval() first"
            )
        selected = routes.float()
        if self.intervention == "mean":
            selected = self.dictionary_mean_weights.unsqueeze(0).expand_as(selected)
        gate_scale = 0.0 if self.intervention == "gates_off" else 1.0
        with torch.autocast(device_type=encoded_a.device.type, enabled=False):
            mixed = torch.einsum(
                "bk,kfd->bfd", selected, self.dictionary_tokens.float()
            )
        tokens = {name: mixed[:, index] for index, name in enumerate(ARTIFACT_FIELD_ORDER)}
        return self.logits_from_tokens(
            encoded_a,
            encoded_b,
            lengths_a,
            lengths_b,
            tokens=tokens,
            gate_scale=gate_scale,
            return_pair_repr=return_pair_repr,
        )

    def forward(
        self, batch: dict[str, torch.Tensor] | None = None, **kwargs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Score routes; emit unreduced masked KL rows when targets are attached."""
        merged: dict[str, torch.Tensor] = {}
        if batch is not None:
            merged.update(batch)
        merged.update(kwargs)
        emb_a, emb_b, lengths_a, lengths_b = unpack_pair_batch(merged, self.input_dim)
        with torch.no_grad():
            encoded_a = self.base.encoder(emb_a, lengths_a)
            encoded_b = self.base.encoder(emb_b, lengths_b)
        if not self.training:
            encoded_a, encoded_b = encoded_a.float(), encoded_b.float()

        target = merged.get(ROUTE_TARGET_KEY)
        route_logits: torch.Tensor | None = None
        if self.mode == "oracle":
            if target is None:
                weights = merged.get(TEMPLATE_KEY)
                if weights is None:
                    raise ValueError(
                        "oracle motif dictionary requires motif_route_targets or motif_weights"
                    )
                target = self.routes_from_weights(weights)
            routes = target.to(device=encoded_a.device, dtype=torch.float32)
        else:
            route_logits = self._route_logits(encoded_a, encoded_b, lengths_a, lengths_b)
            routes = torch.softmax(route_logits, dim=-1)

        # The frozen Stage-I interface is an evaluator in lane R.  It reports
        # task metrics but owns no route-loss gradient and need not retain its
        # much larger trunk activation graph.
        logits_context = (
            torch.no_grad() if self.mode == "sequence" and self.training else nullcontext()
        )
        with logits_context, self._eval_pair_precision(encoded_a.device):
            logits = self.logits_from_routes(
                encoded_a, encoded_b, lengths_a, lengths_b, routes=routes
            )
        output = {"logits": logits, "predicted_routes": routes}
        if target is not None and self.mode == "sequence":
            truth = target.to(device=routes.device, dtype=torch.float32)
            if truth.shape != routes.shape:
                raise ValueError(
                    f"motif_route_targets shape {tuple(truth.shape)} does not match routes "
                    f"{tuple(routes.shape)}"
                )
            assert route_logits is not None
            rows = F.kl_div(
                torch.log_softmax(route_logits.float(), dim=-1),
                truth,
                reduction="none",
            ).sum(dim=-1)
            nonself = merged.get(TEMPLATE_MASK_KEY)
            if nonself is None:
                raise ValueError(
                    f"sequence route targets require nonself mask batch[{TEMPLATE_MASK_KEY!r}]"
                )
            output["route_loss_rows"] = rows * nonself.to(rows).reshape(-1)
        return output


__all__ = [
    "DICTIONARY_INTERVENTIONS",
    "DICTIONARY_MODES",
    "ROUTE_NONEMPTY_KEY",
    "ROUTE_TARGET_KEY",
    "MotifDictionaryConfig",
    "MotifDictionaryRouter",
    "V3_1MotifDictionary",
]
