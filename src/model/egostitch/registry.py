"""Name -> component-class registries and `build_*` helpers (design §3, §8, §12 P3).

`EgoStitchModel.__init__` (`composite.py`) used to hard-wire exactly one
generator, one encoder and one classifier. This module is what makes each
slot selectable by the ``name`` field of its `model.config` section instead:
`GeneratorConfig.name` / `EncoderConfig.name` / `ClassifierConfig.name` look
up a class here, and the matching `build_*` helper turns that class plus the
rest of the section's fields into a constructed component -- the same
translation `EgoStitchModel.__init__` did inline before this module existed.

Only two generators, one encoder and one classifier are registered today
(design §2 Non-goals: this pass ships exactly one real version per component,
behavior-preserving, plus a null generator -- not alternative architectures).
Registering a new component later is adding one dict entry and, if its
constructor needs a translation the existing `build_*` helper does not
already provide, extending that helper -- never a change to `composite.py`'s
selection logic itself.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from src.model.egostitch.classifier import B0V31PairClassifier
from src.model.egostitch.classifier.base import PairClassifier
from src.model.egostitch.config import (
    ClassifierConfig,
    EgoStitchConfig,
    EncoderConfig,
    GeneratorConfig,
)
from src.model.egostitch.encoder import TypedMessagePassingEncoder
from src.model.egostitch.encoder.base import GraphEncoder
from src.model.egostitch.generator import EgoStitchImagineGenerator, NullGenerator
from src.model.egostitch.generator.base import NeighborhoodGenerator
from src.model.egostitch.generator.imagine import FeatureStandardizationMode

GENERATOR_REGISTRY: dict[str, type[NeighborhoodGenerator[Any, Any, Any]]] = {
    "egostitch_imagine": EgoStitchImagineGenerator,
    "null": NullGenerator,
}
"""Generator name -> class. ``egostitch_imagine`` is today's Tokenize-lite +
Imagine + Stitch pipeline (`generator/egostitch.py`); ``null`` imagines
nothing and always returns `None` from `stitch` (`generator/null.py`), which
is what makes the pure pairwise baseline reachable by config alone."""

ENCODER_REGISTRY: dict[str, type[GraphEncoder]] = {
    "ste_typed": TypedMessagePassingEncoder,
}
"""Encoder name -> class. ``ste_typed`` is the typed message-passing port of
today's `STEncoder`, with every dimension read from the graph at runtime."""

CLASSIFIER_REGISTRY: dict[str, type[PairClassifier]] = {
    "b0_v31": B0V31PairClassifier,
}
"""Classifier name -> class. ``b0_v31`` is the mature B0-V3.1 baseline plus
its gated topo-conditioning socket."""


class UnknownComponentError(ValueError):
    """A `model.config` section named a component key that is not registered.

    Deliberately a `ValueError` subclass (not a bare `KeyError`): the message
    always names the offending key and lists every key that *is* registered,
    so a typo in a YAML config fails with an actionable error instead of an
    opaque lookup traceback.
    """


def _resolve(registry: dict[str, type[Any]], name: str, *, kind: str) -> type[Any]:
    """Look up `name` in `registry`, or raise `UnknownComponentError`.

    Deliberately `type[Any]`, not generic over the registry's own component
    bound: each registered class has its own, wider constructor signature
    (`TypedMessagePassingEncoder.__init__` takes `in_dim`/`num_relations`/...,
    none of which exist on the `GraphEncoder` ABC it satisfies), so a `cls:
    type[GraphEncoder]` return here would make every `build_*` helper's own
    constructor call below untypeable. Each `build_*` helper instead `cast`s
    its own constructed *instance* back to the ABC at its return statement --
    the narrowing this function would otherwise buy at the class level is
    unteachable to mypy without also hardcoding the exact per-generator
    constructor signature it will call, at which point the registry stops
    being generic over "any registered class" at all.

    Args:
        registry: One of `GENERATOR_REGISTRY` / `ENCODER_REGISTRY` /
            `CLASSIFIER_REGISTRY`.
        name: The config section's own ``name`` field.
        kind: ``"generator"`` / ``"encoder"`` / ``"classifier"``, used only
            to phrase the error message.

    Returns:
        The registered class.

    Raises:
        UnknownComponentError: If `name` is not a key of `registry`.
    """
    try:
        return registry[name]
    except KeyError:
        available = ", ".join(sorted(registry)) or "(none registered)"
        raise UnknownComponentError(
            f"unknown {kind} name {name!r}; registered {kind}s: {available}"
        ) from None


def resolve_generator_calibration(cfg: GeneratorConfig) -> EgoStitchConfig:
    """Translate a `GeneratorConfig`'s calibration fields into `EgoStitchConfig`.

    `EgoStitchImagineGenerator` keeps its own pinned `EgoStitchConfig`
    defaults (spec Sec 13); `cfg`'s ``n_ground``/``tau_adj``/``tau_div``/
    ``l_gate_pos_weight`` supersede them for this family, exactly as
    `EgoStitchModel.__init__` did inline before the registry existed.

    Exposed separately from `build_generator` (rather than folded into it) so
    a caller can pin the *same* `EgoStitchConfig` object as both the argument
    to `EgoStitchImagineGenerator.__init__` and `EgoStitchModel.generator_cfg`
    -- `composite.py` does exactly this, which is what keeps
    ``model.generator_cfg is model.generator.cfg`` true for the real
    generator while still giving `generator_cfg` a well-defined value (used
    for its ``.n_ground``/``.slots`` fields elsewhere) when the selected
    generator, e.g. `NullGenerator`, carries no `EgoStitchConfig` of its own.

    Args:
        cfg: The `generator:` config section.

    Returns:
        An `EgoStitchConfig` with the four calibration fields overridden.
    """
    return replace(
        EgoStitchConfig(),
        n_ground=cfg.n_ground,
        tau_adj=cfg.tau_adj,
        tau_div=cfg.tau_div,
        l_gate_pos_weight=cfg.l_gate_pos_weight,
    )


def build_generator(
    cfg: GeneratorConfig, *, generator_cfg: EgoStitchConfig
) -> NeighborhoodGenerator[Any, Any, Any]:
    """Build the registered generator named by `cfg.name`.

    Args:
        cfg: The `generator:` config section.
        generator_cfg: The internal `EgoStitchConfig` translation of `cfg`
            (`resolve_generator_calibration`), forwarded to
            `EgoStitchImagineGenerator` unchanged. Ignored by `NullGenerator`,
            which takes no configuration -- there is nothing to imagine,
            nothing to cache, and nothing to stitch (`generator/null.py`).

    Returns:
        The constructed generator, not yet moved to any device.

    Raises:
        UnknownComponentError: If `cfg.name` is not registered.
    """
    cls = _resolve(GENERATOR_REGISTRY, cfg.name, kind="generator")
    if cls is NullGenerator:
        return NullGenerator()
    return cast(
        NeighborhoodGenerator[Any, Any, Any],
        cls(
            generator_cfg,
            feature_standardization=cast(
                FeatureStandardizationMode, cfg.feature_standardization
            ),
            loss_family="egostitch_e2e",
        ),
    )


def build_encoder(
    cfg: EncoderConfig, *, in_dim: int, num_relations: int, d_model: int
) -> GraphEncoder:
    """Build the registered encoder named by `cfg.name`.

    Args:
        cfg: The `encoder:` config section.
        in_dim: `F` -- node feature width, read from the graph the generator
            actually emits (`ImaginedGraph.feature_dim`), never from a
            constant (design §1, §3.1).
        num_relations: `R` -- typed-adjacency relation count, read from
            `ImaginedGraph.num_relations`.
        d_model: Output token/pooled width. This is `ClassifierConfig.d_model`
            -- `EncoderConfig` deliberately carries no `d_model` of its own,
            so the classifier's width is the single source of truth the
            caller must thread through here (design task pin).

    Returns:
        The constructed encoder, not yet moved to any device.

    Raises:
        UnknownComponentError: If `cfg.name` is not registered.
    """
    cls = _resolve(ENCODER_REGISTRY, cfg.name, kind="encoder")
    return cast(
        GraphEncoder,
        cls(
            in_dim=in_dim,
            num_relations=num_relations,
            d_model=d_model,
            dim=cfg.dim,
            layers=cfg.layers,
            w_rel=cfg.w_rel,
        ),
    )


def build_classifier(cfg: ClassifierConfig, *, input_dim: int) -> PairClassifier:
    """Build the registered classifier named by `cfg.name`.

    Args:
        cfg: The `classifier:` config section.
        input_dim: Width `d_in` of the raw per-node token streams the
            classifier's own `encode_tokens` phase consumes (not the
            already-encoded width `forward`'s `PairInputs` carries).

    Returns:
        The constructed classifier, not yet moved to any device.

    Raises:
        UnknownComponentError: If `cfg.name` is not registered.
    """
    cls = _resolve(CLASSIFIER_REGISTRY, cfg.name, kind="classifier")
    return cast(
        PairClassifier,
        cls(
            input_dim=input_dim,
            d_model=cfg.d_model,
            encoder_layers=cfg.encoder_layers,
            n_heads=cfg.n_heads,
            cross_attn_layers=cfg.cross_attn_layers,
            n_inj=cfg.n_inj,
            xattn_heads=cfg.xattn_heads,
            conditioning_ema_decay=cfg.conditioning_ema_decay,
        ),
    )


__all__ = [
    "CLASSIFIER_REGISTRY",
    "ENCODER_REGISTRY",
    "GENERATOR_REGISTRY",
    "UnknownComponentError",
    "build_classifier",
    "build_encoder",
    "build_generator",
    "resolve_generator_calibration",
]
