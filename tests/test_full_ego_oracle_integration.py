"""Narrow integration coverage for the uncapped full-ego oracle diagnostic."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import networkx as nx
import pytest
import src.score_universe as score_universe
import src.train_egostitch as train_egostitch
import yaml  # type: ignore[import-untyped]
from src.model.egostitch.composite import EgoStitchModel
from src.model.egostitch.config import E2EConfig, GeneratorConfig
from src.model.egostitch.generator.full_oracle import FullOracleGenerator
from src.model.egostitch.registry import build_generator, resolve_generator_calibration


def _model_config() -> E2EConfig:
    return E2EConfig.from_mapping(
        {
            "generator": {
                "name": "full_ego_oracle",
                "oracle_truth_source": "g_fit_plus_v_hold",
                "feature_standardization": "row_layernorm",
            },
            "encoder": {"name": "grit_gmt", "w_rel": 0.0},
            "classifier": {"conditioning_mode": "pooled_adapter"},
        }
    )


def test_full_oracle_config_registry_and_arm_are_distinct() -> None:
    cfg = _model_config()
    generator = build_generator(
        cfg.generator,
        generator_cfg=resolve_generator_calibration(cfg.generator),
    )

    assert isinstance(generator, FullOracleGenerator)
    assert generator.graph_dims() == (5, 1)
    assert train_egostitch._e2e_arm_name_from_config(cfg) == "full_ego_oracle"


def test_non_oracle_cannot_request_v_hold_truth() -> None:
    with pytest.raises(ValueError, match="requires name='oracle_struct'"):
        GeneratorConfig(name="egostitch_imagine", oracle_truth_source="g_fit_plus_v_hold")


def test_scoring_installs_full_truth_graph_without_slot_table() -> None:
    model = EgoStitchModel(_model_config())
    truth = nx.Graph([("u", "w")])

    score_universe._install_oracle_context(model, ["u", "v"], truth_graph=truth)

    assert isinstance(model.generator, FullOracleGenerator)
    assert model.generator._node_ids == ("u", "v")
    assert model.generator._graph is not None
    assert set(model.generator._graph) == {"u", "v", "w"}
    assert model.generator._graph.degree("v") == 0


def test_training_full_oracle_is_diagnostic_only() -> None:
    model = EgoStitchModel(_model_config())

    with pytest.raises(RuntimeError, match="requires --run-kind diagnostic"):
        train_egostitch._install_oracle_context(
            model,
            cast(Any, SimpleNamespace()),  # rejected before training-data access
            run_kind="formal",
        )


def test_full_oracle_skips_inapplicable_initial_slot_health_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = EgoStitchModel(_model_config())

    def _unexpected_validation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("full oracle has no learned slot geometry to validate")

    monkeypatch.setattr(train_egostitch, "_validate_epoch", _unexpected_validation)
    validation_events: list[tuple[str, int | None, int]] = []

    report = train_egostitch._enforce_e2e_initial_slot_health(
        model,
        cast(Any, SimpleNamespace(val_pairs=[("u", "v")])),
        cast(Any, SimpleNamespace()),
        edge_batch=1,
        topk_fraction=0.1,
        token_table=None,
        token_node_index=None,
        validation_event_callback=lambda kind, epoch, step: validation_events.append(
            (kind, epoch, step)
        ),
    )

    assert report == {}
    assert validation_events == []


def test_runnable_config_uses_logical_128_training_and_singleton_scoring() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "egostitch_e2e_v3_full_ego_oracle_grit_pooled_breadth_first.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    run_cfg = train_egostitch.load_config(path)
    cfg = E2EConfig.from_mapping(payload["model"]["config"])

    assert cfg.generator.name == "full_ego_oracle"
    assert cfg.generator.oracle_truth_source == "g_fit_plus_v_hold"
    assert cfg.encoder.name == "grit_gmt"
    assert cfg.encoder.w_rel == 0.0
    assert cfg.classifier.conditioning_mode == "pooled_adapter"
    assert payload["training"]["phase_a_fraction"] == 0.0
    assert run_cfg.data.edge_batch == 16
    assert run_cfg.optim.gradient_accumulation_steps == 8
    assert run_cfg.data.edge_batch * run_cfg.optim.gradient_accumulation_steps == 128
    assert run_cfg.optim.epochs == 10
    assert score_universe._full_oracle_score_batch_specs(3) == [
        ([0], [(0, 0)]),
        ([1], [(1, 0)]),
        ([2], [(2, 0)]),
    ]
