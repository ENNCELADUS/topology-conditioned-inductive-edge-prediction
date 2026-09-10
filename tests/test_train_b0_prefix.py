"""train_b0 dispatch for the v3_1_prefix family."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch
from src.eval.edge_metrics import EdgeMetrics
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.model.egostitch.classifier.prefix import V3_1Prefix
from src.train_b0 import (
    MODEL_FAMILIES,
    TrainResult,
    _base_loss_kwargs,
    _build_optimizer,
    _init_prefix_from_loader,
    _run_metadata,
    build_model,
    is_v3_1_family,
    load_config,
    resolve_model_kwargs,
)

from tests.test_prefix_model import _pair_batch, _tiny_base_config
from tests.test_train_b0 import _write_yaml_config


def _base_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    base = V3_1(**_tiny_base_config())
    payload = {
        "model_state": base.state_dict(),
        "model_family": "v3_1",
        "model_config": _tiny_base_config(),
        "epoch": 1,
        "val_metrics": {},
        "seed": 0,
        "config": {},
    }
    path = tmp_path / "best.pt"
    torch.save(payload, path)
    return path


def _prefix_yaml(tmp_path: Path, conditioning: str = "pair") -> Path:
    config_path = tmp_path / "prefix.yaml"
    _write_yaml_config(
        config_path,
        {
            "model": {
                "family": "v3_1_prefix",
                "config": {
                    "prefix": {
                        "tokens": 3,
                        "rank": 2,
                        "conditioning": conditioning,
                        "bottleneck": 6,
                        "base_checkpoint": str(_base_checkpoint(tmp_path)),
                    }
                },
            }
        },
    )
    return config_path


def test_family_is_registered_and_grouped_with_v3_1() -> None:
    assert "v3_1_prefix" in MODEL_FAMILIES
    assert is_v3_1_family("v3_1") and is_v3_1_family("v3_1_prefix")
    assert not is_v3_1_family("f0_mlp")


def test_resolve_embeds_base_config_and_provenance(tmp_path: Path) -> None:
    cfg = load_config(_prefix_yaml(tmp_path))
    kwargs = resolve_model_kwargs(cfg.model)
    assert kwargs["base"] == _tiny_base_config()
    prefix = kwargs["prefix"]
    assert isinstance(prefix, dict)
    assert prefix["base_checkpoint"].endswith("best.pt")
    sha256 = prefix["base_checkpoint_sha256"]
    assert isinstance(sha256, str) and len(sha256) == 64
    json.dumps(kwargs)  # checkpoint-embeddable


def test_run_metadata_records_the_frozen_base_provenance(tmp_path: Path) -> None:
    """`run_metadata.json` has a fixed key list that omits ``model_config``.

    Review P2: without this block a published prefix run's only record of which
    base it froze lives inside ``best.pt``. Recorded, never verified.
    """
    cfg = load_config(_prefix_yaml(tmp_path))
    kwargs = resolve_model_kwargs(cfg.model)
    metrics = EdgeMetrics(
        auroc=0.9,
        auprc=0.85,
        accuracy=0.8,
        sensitivity=0.7,
        specificity=0.9,
        precision=0.75,
        recall=0.7,
        f1=0.72,
        mcc=0.5,
        ece=0.05,
        brier=0.1,
        threshold=0.5,
        n_pos=10,
        n_neg=10,
    )
    state: dict[str, torch.Tensor] = {"w": torch.zeros(2)}
    result = TrainResult(
        best_state_dict=state,
        best_epoch=2,
        best_val_metrics=metrics,
        last_state_dict=state,
        last_epoch=3,
        last_val_metrics=metrics,
        history=[],
        stopped_early=False,
    )

    metadata = _run_metadata(result, cfg, kwargs, {}, {})

    prefix_base = metadata["prefix_base"]
    assert isinstance(prefix_base, dict)
    assert str(prefix_base["checkpoint"]).endswith("best.pt")
    sha256 = prefix_base["sha256"]
    assert isinstance(sha256, str) and len(sha256) == 64
    assert sha256 == cast(dict[str, object], kwargs["prefix"])["base_checkpoint_sha256"]
    json.dumps(metadata)


def test_base_loss_kwargs_unwraps_the_nested_base_for_prefix(tmp_path: Path) -> None:
    """Verify the base config, not the top-level wrapper, is unwrapped.

    `_base_loss_kwargs` must read the frozen base's config, not the
    ``{"base": ..., "prefix": ...}`` wrapper `resolve_model_kwargs` returns for
    ``v3_1_prefix`` -- a plain ``.get("positive_weight", 1.0)`` on the wrapper
    always misses and silently falls back to the default (review round 1 P2).
    """
    cfg = load_config(_prefix_yaml(tmp_path))
    loss_kwargs = _base_loss_kwargs(cfg.model)
    assert loss_kwargs["positive_weight"] == _tiny_base_config()["positive_weight"] == 5.0
    assert loss_kwargs["label_smoothing"] == _tiny_base_config()["label_smoothing"]


def test_build_model_loads_and_freezes_the_base(tmp_path: Path) -> None:
    cfg = load_config(_prefix_yaml(tmp_path))
    model = build_model(cfg)
    assert isinstance(model, V3_1Prefix)
    saved = torch.load(cfg.model.config["prefix"]["base_checkpoint"], map_location="cpu")  # type: ignore[index]
    for name, tensor in saved["model_state"].items():
        assert torch.equal(model.base.state_dict()[name], tensor)
    assert all(not p.requires_grad for p in model.base.parameters())
    optimizer = _build_optimizer(model, cfg)
    ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert ids == {id(p) for p in model.prefix_parameters()}


def test_missing_base_checkpoint_key_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    _write_yaml_config(
        config_path, {"model": {"family": "v3_1_prefix", "config": {"prefix": {"tokens": 3}}}}
    )
    try:
        resolve_model_kwargs(load_config(config_path).model)
    except ValueError as err:
        assert "base_checkpoint" in str(err)
    else:
        raise AssertionError("prefix.base_checkpoint is required")


def test_extra_model_config_keys_raise(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_extra.yaml"
    _write_yaml_config(
        config_path,
        {
            "model": {
                "family": "v3_1_prefix",
                "config": {
                    "prefix": {
                        "tokens": 3,
                        "rank": 2,
                        "conditioning": "pair",
                        "bottleneck": 6,
                        "base_checkpoint": str(_base_checkpoint(tmp_path)),
                    },
                    "extra_key": 1,
                },
            }
        },
    )
    with pytest.raises(ValueError, match="prefix"):
        resolve_model_kwargs(load_config(config_path).model)


def test_non_v3_1_base_checkpoint_raises(tmp_path: Path) -> None:
    torch.manual_seed(0)
    base = V3_1(**_tiny_base_config())
    payload = {
        "model_state": base.state_dict(),
        "model_family": "egostitch_e2e",
        "model_config": _tiny_base_config(),
        "epoch": 1,
        "val_metrics": {},
        "seed": 0,
        "config": {},
    }
    path = tmp_path / "teacher.pt"
    torch.save(payload, path)
    config_path = tmp_path / "bad_family.yaml"
    _write_yaml_config(
        config_path,
        {
            "model": {
                "family": "v3_1_prefix",
                "config": {
                    "prefix": {
                        "tokens": 3,
                        "rank": 2,
                        "conditioning": "pair",
                        "bottleneck": 6,
                        "base_checkpoint": str(path),
                    }
                },
            }
        },
    )
    with pytest.raises(ValueError, match="v3_1"):
        resolve_model_kwargs(load_config(config_path).model)


def _tiny_prefix_model() -> V3_1Prefix:
    return V3_1Prefix(
        base=_tiny_base_config(),
        prefix={"tokens": 3, "rank": 2, "conditioning": "pair", "bottleneck": 6},
    )


def test_init_prefix_from_loader_casts_a_bfloat16_batch() -> None:
    model = _tiny_prefix_model()
    before = model.generator.p0[0].clone()
    batch = _pair_batch()
    batch["emb_a"] = batch["emb_a"].to(torch.bfloat16)
    batch["emb_b"] = batch["emb_b"].to(torch.bfloat16)
    _init_prefix_from_loader(model, [batch], seed=0)
    after = model.generator.p0[0]
    assert not torch.equal(before, after)
    assert torch.isfinite(after).all()


def test_init_prefix_from_loader_raises_on_an_empty_loader() -> None:
    model = _tiny_prefix_model()
    with pytest.raises(ValueError, match="non-empty"):
        _init_prefix_from_loader(model, [], seed=0)
