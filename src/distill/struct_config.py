"""`StructConfig`: the structural-stream topology-loss knobs.

Consumed by `src.train_b0` as the optional top-level ``struct:`` config
section (spec: ``docs/superpowers/specs/2026-09-07-structural-stream-topology-losses-design.md``).
Each optimizer step forwards one sampled training subgraph and supervises its
output adjacency with the weighted terms named in ``weights``. Any non-negative
weight combination is legal; the arm name is the sorted list of nonzero keys.
An absent block (``Config.struct is None``) leaves training bit-identical.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields

WEIGHT_KEYS: tuple[str, ...] = ("bce", "gs", "rd", "deg_mmd", "rank", "degree", "motif")
KINDS: tuple[str, ...] = ("bfs", "motif", "bridge")
_INT_FIELDS = frozenset({"nodes", "background_nodes", "mmd_bins", "val_subgraphs"})
_FLOAT_FIELDS = frozenset({"rank_margin", "rank_temperature", "huber_delta", "mmd_sigma"})


def _default_mix() -> dict[str, float]:
    return {"bfs": 0.5, "motif": 0.25, "bridge": 0.25}


def _default_weights() -> dict[str, float]:
    return dict.fromkeys(WEIGHT_KEYS, 0.0)


@dataclass(frozen=True)
class StructConfig:
    """Structural-stream configuration.

    Attributes:
        nodes: Subgraph size ``n`` (local budget is ``nodes - background_nodes``).
        background_nodes: Uniformly drawn nodes appended to every subgraph.
        mix: Sampling share per subgraph kind (``bfs``, ``motif``, ``bridge``); sums to 1.
        subgraphs_per_epoch: Plan length; ``None`` means one subgraph per global
            optimizer step.
        weights: Non-negative weight per term key in ``WEIGHT_KEYS``; at least one nonzero.
        rank_margin: Margin ``m`` of the neighbour-ranking term.
        rank_temperature: Temperature ``T`` of the neighbour-ranking term.
        huber_delta: SmoothL1 transition point shared by ``rd``, ``degree`` and ``motif``.
        mmd_sigma: Gaussian width of the degree soft histogram and of its TV kernel.
        mmd_bins: Number of degree histogram centres on ``[0, n-1]``.
        val_subgraphs: Fixed V_val diagnostic subgraph count.
    """

    nodes: int = 40
    background_nodes: int = 8
    mix: dict[str, float] = field(default_factory=_default_mix)
    subgraphs_per_epoch: int | None = None
    weights: dict[str, float] = field(default_factory=_default_weights)
    rank_margin: float = 0.1
    rank_temperature: float = 1.0
    huber_delta: float = 1.0
    mmd_sigma: float = 1.25
    mmd_bins: int = 48
    val_subgraphs: int = 32

    def __post_init__(self) -> None:
        """Validate ranges, the kind mix, and the weight pattern.

        Raises:
            ValueError: On any illegal value.
        """
        if self.nodes < 4:
            raise ValueError(f"struct.nodes must be >= 4, got {self.nodes}")
        if not 0 <= self.background_nodes < self.nodes:
            raise ValueError(
                f"struct.background_nodes must be in [0, nodes), got {self.background_nodes}"
            )
        if set(self.mix) != set(KINDS):
            raise ValueError(f"struct.mix must name exactly {list(KINDS)}, got {sorted(self.mix)}")
        shares = {kind: float(self.mix[kind]) for kind in KINDS}
        if any(share < 0.0 for share in shares.values()) or abs(sum(shares.values()) - 1.0) > 1e-6:
            raise ValueError(f"struct.mix shares must be non-negative and sum to 1, got {self.mix}")
        if self.subgraphs_per_epoch is not None and self.subgraphs_per_epoch < 1:
            raise ValueError("struct.subgraphs_per_epoch must be positive or null")
        unknown = sorted(set(self.weights) - set(WEIGHT_KEYS))
        if unknown:
            raise ValueError(f"unknown struct weight keys: {unknown}")
        normalised = {key: float(self.weights.get(key, 0.0)) for key in WEIGHT_KEYS}
        for key, value in normalised.items():
            if value < 0.0:
                raise ValueError(f"struct.weights.{key} must be non-negative")
        # Frozen dataclass: fill the missing keys through object.__setattr__.
        object.__setattr__(self, "weights", normalised)
        object.__setattr__(self, "mix", shares)
        if not self.active_weights:
            raise ValueError("struct.weights must have at least one nonzero term")
        if self.rank_margin < 0.0:
            raise ValueError("struct.rank_margin must be non-negative")
        if self.rank_temperature <= 0.0:
            raise ValueError("struct.rank_temperature must be positive")
        if self.huber_delta <= 0.0:
            raise ValueError("struct.huber_delta must be positive")
        if self.mmd_sigma <= 0.0:
            raise ValueError("struct.mmd_sigma must be positive")
        if self.mmd_bins < 2:
            raise ValueError("struct.mmd_bins must be >= 2")
        if self.val_subgraphs < 0:
            raise ValueError("struct.val_subgraphs must be non-negative")

    @property
    def active_weights(self) -> dict[str, float]:
        """Nonzero weights in ``WEIGHT_KEYS`` order."""
        return {
            key: float(self.weights[key]) for key in WEIGHT_KEYS if self.weights.get(key, 0.0) > 0.0
        }

    @property
    def arm(self) -> str:
        """Arm name: the sorted nonzero weight keys joined by ``+``."""
        return "+".join(sorted(self.active_weights))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> StructConfig:
        """Build the ``struct:`` section from a YAML mapping.

        Raises:
            ValueError: On unknown keys or ill-typed values.
        """
        known = {spec.name for spec in fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"unknown struct config keys: {unknown}")
        kwargs: dict[str, object] = {}
        for spec in fields(cls):
            if spec.name not in mapping:
                continue
            raw = mapping[spec.name]
            if spec.name in {"mix", "weights"}:
                if not isinstance(raw, Mapping):
                    raise ValueError(f"struct.{spec.name} must be a mapping")
                kwargs[spec.name] = {
                    str(key): _number(value, f"struct.{spec.name}.{key}")
                    for key, value in raw.items()
                }
            elif spec.name == "subgraphs_per_epoch":
                kwargs[spec.name] = (
                    None if raw is None else int(_number(raw, "struct.subgraphs_per_epoch"))
                )
            elif spec.name in _INT_FIELDS:
                kwargs[spec.name] = int(_number(raw, f"struct.{spec.name}"))
            elif spec.name in _FLOAT_FIELDS:
                kwargs[spec.name] = _number(raw, f"struct.{spec.name}")
        return cls(**kwargs)  # type: ignore[arg-type]


def _number(raw: object, label: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(raw)


__all__ = ["KINDS", "WEIGHT_KEYS", "StructConfig"]
