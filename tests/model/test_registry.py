"""Tests for the component registries and `build_*` helpers (design §3, §8, §12 P3).

Three concerns, one per registry: the registered name resolves to the exact
class `composite.py` expects, an unknown name fails with a listing error
(never a bare `KeyError`), and the object a `build_*` helper returns actually
satisfies its component's ABC -- callable through the same protocol methods
`EgoStitchModel` drives (`encode_node`/`stitch`/`auxiliary_losses`,
`forward`/`auxiliary_losses`, `encode_tokens`/`forward`).
"""

from __future__ import annotations

import pytest
import torch
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
from src.model.egostitch.graph import ImaginedGraph, PairInputs
from src.model.egostitch.registry import (
    CLASSIFIER_REGISTRY,
    ENCODER_REGISTRY,
    GENERATOR_REGISTRY,
    UnknownComponentError,
    build_classifier,
    build_encoder,
    build_generator,
    resolve_generator_calibration,
)

# --------------------------------------------------------------------------- registry contents


def test_generator_registry_has_exactly_the_five_registered_keys() -> None:
    """`GENERATOR_REGISTRY` resolves the real, null, and three oracle generators.

    (Design §8; `oracle_struct` added by the 2026-08-04 oracle-scaffold wave,
    `full_ego_features` by the Gate A set-student wave.)
    """
    from src.model.egostitch.generator.full_oracle import (
        FullEgoFeaturesGenerator,
        FullOracleGenerator,
    )
    from src.model.egostitch.generator.oracle import OracleStructGenerator

    assert {
        "egostitch_imagine": EgoStitchImagineGenerator,
        "null": NullGenerator,
        "oracle_struct": OracleStructGenerator,
        "full_ego_oracle": FullOracleGenerator,
        "full_ego_features": FullEgoFeaturesGenerator,
    } == GENERATOR_REGISTRY


def test_encoder_registry_has_exactly_the_two_registered_keys() -> None:
    """`ENCODER_REGISTRY` resolves `ste_typed` and `grit_gmt` (2026-08-04 wave)."""
    from src.model.egostitch.encoder.grit_gmt import GritGmtEncoder

    assert {
        "ste_typed": TypedMessagePassingEncoder,
        "grit_gmt": GritGmtEncoder,
    } == ENCODER_REGISTRY


def test_classifier_registry_has_exactly_the_one_registered_key() -> None:
    """`CLASSIFIER_REGISTRY` resolves `b0_v31`."""
    assert {"b0_v31": B0V31PairClassifier} == CLASSIFIER_REGISTRY


# --------------------------------------------------------------------------- unknown-name errors


def test_build_generator_unknown_name_lists_registered_generators() -> None:
    cfg = GeneratorConfig(name="not_a_real_generator")
    with pytest.raises(UnknownComponentError) as excinfo:
        build_generator(cfg, generator_cfg=EgoStitchConfig())
    message = str(excinfo.value)
    assert "not_a_real_generator" in message
    assert "egostitch_imagine" in message
    assert "null" in message


def test_build_encoder_unknown_name_lists_registered_encoders() -> None:
    cfg = EncoderConfig(name="not_a_real_encoder")
    with pytest.raises(UnknownComponentError) as excinfo:
        build_encoder(cfg, in_dim=11, num_relations=4, d_model=16)
    message = str(excinfo.value)
    assert "not_a_real_encoder" in message
    assert "ste_typed" in message


def test_build_classifier_unknown_name_lists_registered_classifiers() -> None:
    cfg = ClassifierConfig(name="not_a_real_classifier")
    with pytest.raises(UnknownComponentError) as excinfo:
        build_classifier(cfg, input_dim=16)
    message = str(excinfo.value)
    assert "not_a_real_classifier" in message
    assert "b0_v31" in message


def test_unknown_component_error_is_a_value_error_not_a_key_error() -> None:
    """The unknown-name failure must never surface as a bare `KeyError`."""
    with pytest.raises(ValueError):
        build_classifier(ClassifierConfig(name="nope"), input_dim=16)


# --------------------------------------------------------------------------- build_generator


def test_build_generator_egostitch_imagine_satisfies_the_generator_protocol() -> None:
    """`build_generator` wires the calibration fields through and returns a real generator."""
    cfg = GeneratorConfig(
        name="egostitch_imagine",
        n_ground=7,
        tau_adj=0.4,
        tau_div=0.3,
        l_gate_pos_weight=5.0,
        feature_standardization="row_layernorm",
    )
    generator_cfg = resolve_generator_calibration(cfg)
    generator = build_generator(cfg, generator_cfg=generator_cfg)

    assert isinstance(generator, NeighborhoodGenerator)
    assert isinstance(generator, EgoStitchImagineGenerator)
    # `build_generator` forwards the *same* `EgoStitchConfig` object -- the
    # composite relies on this identity for `EgoStitchModel.generator_cfg`.
    assert generator.cfg is generator_cfg
    assert generator.cfg.n_ground == 7
    assert generator.cfg.tau_adj == 0.4
    assert generator.cfg.tau_div == 0.3
    assert generator.cfg.l_gate_pos_weight == 5.0

    # Protocol-level smoke test: encode_node -> stitch -> a real graph.
    x = torch.randn(2, generator_cfg.input_dim)
    ground = torch.randn(2, cfg.n_ground, generator_cfg.input_dim)
    state = generator.encode_node(x, ground)
    graph = generator.stitch(state, state, torch.ones(2, dtype=torch.bool))
    assert isinstance(graph, ImaginedGraph)
    assert graph.feature_dim > 0
    assert graph.num_relations > 0


def test_build_generator_null_ignores_generator_cfg_and_takes_no_parameters() -> None:
    """`build_generator` special-cases `null`: no configuration reaches it."""
    cfg = GeneratorConfig(name="null")
    generator = build_generator(cfg, generator_cfg=resolve_generator_calibration(cfg))

    assert isinstance(generator, NeighborhoodGenerator)
    assert isinstance(generator, NullGenerator)
    assert sum(p.numel() for p in generator.parameters()) == 0

    x = torch.randn(3, 16)
    ground = torch.randn(3, 5, 16)
    state_a = generator.encode_node(x, ground)
    state_b = generator.encode_node(x, ground)
    graph = generator.stitch(state_a, state_b, torch.zeros(3, dtype=torch.bool))
    assert graph is None
    assert generator.auxiliary_losses(graph, {}) == {}


def test_resolve_generator_calibration_overrides_only_the_four_calibration_fields() -> None:
    """Every other `EgoStitchConfig` field keeps its own pinned default."""
    cfg = GeneratorConfig(n_ground=42, tau_adj=0.2, tau_div=-0.5, l_gate_pos_weight=3.0)
    generator_cfg = resolve_generator_calibration(cfg)
    default = EgoStitchConfig()

    assert generator_cfg.n_ground == 42
    assert generator_cfg.tau_adj == 0.2
    assert generator_cfg.tau_div == -0.5
    assert generator_cfg.l_gate_pos_weight == 3.0
    assert generator_cfg.input_dim == default.input_dim
    assert generator_cfg.slots == default.slots
    assert generator_cfg.w_rel == default.w_rel


# --------------------------------------------------------------------------- build_encoder


def test_build_encoder_wires_runtime_graph_dims_not_constants() -> None:
    """`build_encoder` sizes the encoder from the caller-supplied `in_dim`/`num_relations`.

    Design §1/§3.1: the encoder's input dims must come from the graph the
    generator actually emits, never a module constant -- `build_encoder`'s
    `in_dim`/`num_relations` parameters are exactly that seam.
    """
    cfg = EncoderConfig(name="ste_typed", dim=8, layers=2, w_rel=0.0)
    encoder = build_encoder(cfg, in_dim=13, num_relations=5, d_model=24)

    assert isinstance(encoder, GraphEncoder)
    assert isinstance(encoder, TypedMessagePassingEncoder)
    assert encoder.in_dim == 13
    assert encoder.num_relations == 5
    assert encoder.out_dim == 24
    assert encoder.rel_head is None  # w_rel == 0.0 omits the relational head

    batch, nodes = 2, 6
    graph = ImaginedGraph(
        x=torch.randn(batch, nodes, 13),
        adj=torch.rand(batch, 5, nodes, nodes),
        mask=torch.ones(batch, nodes),
        aux={},
    )
    embedding = encoder(graph)
    assert embedding.tokens.shape == (batch, nodes, 24)
    assert embedding.pooled.shape == (batch, 24)


def test_build_encoder_d_model_comes_from_the_caller_not_encoder_config() -> None:
    """`EncoderConfig` carries no `d_model` of its own (task pin): the caller supplies it."""
    assert not hasattr(EncoderConfig(), "d_model")
    encoder = build_encoder(EncoderConfig(), in_dim=11, num_relations=4, d_model=64)
    assert encoder.out_dim == 64


# --------------------------------------------------------------------------- build_classifier


def test_build_classifier_satisfies_the_classifier_protocol() -> None:
    cfg = ClassifierConfig(
        d_model=16, encoder_layers=1, cross_attn_layers=1, n_heads=2, n_inj=1, xattn_heads=2
    )
    classifier = build_classifier(cfg, input_dim=32)

    assert isinstance(classifier, PairClassifier)
    assert isinstance(classifier, B0V31PairClassifier)

    b, t = 3, 4
    tokens_a = classifier.encode_tokens(torch.randn(b, t, 32), torch.full((b,), t))
    tokens_b = classifier.encode_tokens(torch.randn(b, t, 32), torch.full((b,), t))
    pair = PairInputs(
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        len_a=torch.full((b,), t),
        len_b=torch.full((b,), t),
    )
    logits = classifier(pair, None)
    assert logits.shape == (b,)
