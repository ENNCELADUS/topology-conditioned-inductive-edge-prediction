"""Trainer plumbing for the Stage II coordinate generator (v3_1_coord_gen)."""

from __future__ import annotations

import functools
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from src.data.struct_coords import COORD_DIM
from src.model.egostitch.classifier.coord_gen import V3_1CoordGen
from src.model.egostitch.classifier.topo_prompt import V3_1TopoPrompt
from src.train_b0 import (
    COORD_GEN_FAMILY,
    MODEL_FAMILIES,
    TopoPromptRows,
    TrainResult,
    ValidationOutcome,
    _base_loss_kwargs,
    _build_optimizer,
    _coordinate_fit_metrics,
    _evaluate_distributed,
    _run_metadata,
    build_model,
    config_to_dict,
    is_v3_1_family,
    load_config,
    resolve_model_kwargs,
    train_ddp_loop,
)

from tests.test_prefix_model import _tiny_base_config
from tests.test_train_b0 import _constant_metrics, _tiny_config, _write_yaml_config
from tests.test_train_b0_prefix import _base_checkpoint
from tests.test_train_b0_topo_prompt import _prompt_batches, _tiny_graph


def _reader_checkpoint(tmp_path: Path, *, with_stats: bool = True) -> Path:
    torch.manual_seed(0)
    reader = V3_1TopoPrompt(
        base=_tiny_base_config(),
        topo_prompt={"trainable": "all", "width": 8, "slots_per_field": 1, "field_mask_prob": 0.0},
    )
    if with_stats:
        reader.generator.set_coord_stats(
            torch.linspace(-1.0, 1.0, COORD_DIM), torch.linspace(0.5, 2.0, COORD_DIM), 9
        )
    with torch.no_grad():
        reader.generator.gates.fill_(0.3)
    payload = {
        "model_state": reader.state_dict(),
        "model_family": "v3_1_topo_prompt",
        "model_config": {"base": _tiny_base_config(), "topo_prompt": reader.cfg.to_dict()},
        "epoch": 1,
        "val_metrics": {},
        "seed": 0,
        "config": {},
    }
    path = tmp_path / ("reader.pt" if with_stats else "reader_nostats.pt")
    torch.save(payload, path)
    return path


def _gen_yaml(tmp_path: Path, **block: object) -> Path:
    config_path = tmp_path / "gen.yaml"
    full_block: dict[str, object] = {
        "reader_checkpoint": str(_reader_checkpoint(tmp_path)),
        "hidden": 16,
        "layers": 1,
        "dropout": 0.0,
    }
    full_block.update(block)
    _write_yaml_config(
        config_path, {"model": {"family": COORD_GEN_FAMILY, "config": {"coord_gen": full_block}}}
    )
    return config_path


def test_family_is_registered_and_grouped_with_v3_1() -> None:
    assert COORD_GEN_FAMILY in MODEL_FAMILIES
    assert is_v3_1_family(COORD_GEN_FAMILY)


def test_resolve_and_build_from_a_reader_checkpoint(tmp_path: Path) -> None:
    cfg = load_config(_gen_yaml(tmp_path))
    kwargs = resolve_model_kwargs(cfg.model)
    reader = cast(dict[str, object], kwargs["reader"])
    assert reader["base"] == _tiny_base_config()
    block = cast(dict[str, object], kwargs["coord_gen"])
    assert block["reader_checkpoint_sha256"] is not None and block["w_kd"] == 0.1
    assert _base_loss_kwargs(cfg.model)["positive_weight"] == 5.0
    model = build_model(cfg)
    assert isinstance(model, V3_1CoordGen)
    assert float(model.reader.generator.coord_count) == 9.0
    assert float(model.reader.generator.gates.mean()) == pytest.approx(0.3)
    assert all(not param.requires_grad for param in model.reader.parameters())
    optimizer = _build_optimizer(model, cfg)
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(
        list(model.generator.parameters())
    )


def test_resolution_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    _write_yaml_config(missing, {"model": {"family": COORD_GEN_FAMILY, "config": {}}})
    with pytest.raises(ValueError, match="coord_gen is required"):
        resolve_model_kwargs(load_config(missing).model)
    sibling = tmp_path / "sibling.yaml"
    _write_yaml_config(
        sibling,
        {
            "model": {
                "family": COORD_GEN_FAMILY,
                "config": {"coord_gen": {"reader_checkpoint": "x.pt"}, "base": {}},
            }
        },
    )
    with pytest.raises(ValueError, match="accepts only"):
        resolve_model_kwargs(load_config(sibling).model)
    no_reader = tmp_path / "no_reader.yaml"
    _write_yaml_config(
        no_reader, {"model": {"family": COORD_GEN_FAMILY, "config": {"coord_gen": {}}}}
    )
    with pytest.raises(ValueError, match="reader_checkpoint is required"):
        resolve_model_kwargs(load_config(no_reader).model)
    with pytest.raises(ValueError, match="v3_1_topo_prompt"):
        resolve_model_kwargs(
            load_config(
                _gen_yaml(tmp_path, reader_checkpoint=str(_base_checkpoint(tmp_path)))
            ).model
        )
    no_stats = _gen_yaml(
        tmp_path, reader_checkpoint=str(_reader_checkpoint(tmp_path, with_stats=False))
    )
    with pytest.raises(ValueError, match="no coordinate statistics"):
        build_model(load_config(no_stats))


def test_run_metadata_records_the_reader_and_weights(tmp_path: Path) -> None:
    cfg = load_config(_gen_yaml(tmp_path))
    kwargs = resolve_model_kwargs(cfg.model)
    state = build_model(cfg).state_dict()
    result = TrainResult(
        best_state_dict=state,
        last_state_dict=state,
        best_epoch=1,
        last_epoch=1,
        best_val_metrics=_constant_metrics(),
        last_val_metrics=_constant_metrics(),
        history=[],
        stopped_early=False,
        stop_epoch=None,
    )
    metadata = _run_metadata(result, cfg, kwargs, {}, config_to_dict(cfg))
    block = cast(dict[str, object], metadata["coord_gen"])
    assert block["reader_checkpoint"] == str(tmp_path / "reader.pt")
    assert block["sha256"] is not None
    assert cast(dict[str, float], block["weights"])["kd"] == 0.1
    assert "run_kind" not in metadata
    assert metadata["arm"] == COORD_GEN_FAMILY
    json.dumps(metadata)


def _rows_and_batches() -> tuple[
    TopoPromptRows, list[dict[str, torch.Tensor]], list[tuple[str, str]]
]:
    train, val, nodes = _tiny_graph()
    rows_per_batch, n_batches = 4, 3
    n_rows = rows_per_batch * n_batches
    rng = np.random.default_rng(0)
    train_pairs = [
        (nodes[int(i)], nodes[int(j)]) for i, j in rng.integers(len(nodes), size=(n_rows, 2))
    ]
    val_pairs = [("v0", "v1"), ("v1", "v3"), ("v2", "v4"), ("v0", "v3")]
    rows = TopoPromptRows(
        train_graph=train,
        train_pairs=train_pairs,
        stats_rows=np.arange(n_rows),
        val_graph=val,
        val_cls_pairs=val_pairs,
        universe_pairs=val_pairs,
        device=torch.device("cpu"),
    )
    return rows, _prompt_batches(n_batches, rows_per_batch), val_pairs


def test_coordinate_fit_metrics_are_collective_and_bounded(tmp_path: Path) -> None:
    rows, _, val_pairs = _rows_and_batches()
    model = build_model(load_config(_gen_yaml(tmp_path)))
    accelerator = Accelerator(cpu=True)
    val_batches = _prompt_batches(1, len(val_pairs))
    model.train()
    metrics = _coordinate_fit_metrics(model, val_batches, accelerator, attach=rows.attach_val)
    assert model.training  # restored
    assert set(metrics) == {
        "val_coord_r2_endpoint",
        "val_coord_r2_relation",
        "val_coord_r2_context",
        "val_coord_dist_acc",
        "val_coord_loss",
        "val_kd_loss",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0.0 <= metrics["val_coord_dist_acc"] <= 1.0
    assert max(metrics[f"val_coord_r2_{f}"] for f in ("endpoint", "relation", "context")) <= 1.0
    again = _coordinate_fit_metrics(model, val_batches, accelerator, attach=rows.attach_val)
    assert again == metrics
    plain = V3_1TopoPrompt(base=_tiny_base_config(), topo_prompt={"trainable": "all", "width": 8})
    with pytest.raises(TypeError, match="V3_1CoordGen"):
        _coordinate_fit_metrics(plain, val_batches, accelerator, attach=rows.attach_val)


def test_ddp_loop_trains_only_the_generator_and_logs_the_fit(tmp_path: Path) -> None:
    rows, batches, val_pairs = _rows_and_batches()
    model = build_model(load_config(_gen_yaml(tmp_path)))
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    val_batches = _prompt_batches(1, len(val_pairs))
    cfg = replace(load_config(_gen_yaml(tmp_path)), output_dir=tmp_path)
    cfg = replace(cfg, optim=_tiny_config(epochs=1).optim, eval=_tiny_config(epochs=1).eval)
    accelerator = Accelerator(cpu=True)

    def evaluate(
        model_: torch.nn.Module, loader: object, accelerator_: Accelerator
    ) -> ValidationOutcome:
        return _evaluate_distributed(
            model_,
            cast(list[dict[str, torch.Tensor]], loader),
            accelerator_,
            expected_row_ids=np.arange(len(val_pairs)),
            attach=rows.attach_val,
            diagnostics_fn=functools.partial(_coordinate_fit_metrics, attach=rows.attach_val),
        )

    result = train_ddp_loop(
        model,
        lambda epoch: batches,
        val_batches,
        cfg,
        accelerator,
        warmup_steps=1,
        artifact_dir=tmp_path,
        evaluate_fn=evaluate,
        topo_rows=rows,
    )
    after = model.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before if key.startswith("reader."))
    assert any(
        not torch.equal(before[key], after[key]) for key in before if key.startswith("generator.")
    )
    last = result.history[-1]
    assert last["epoch"] == 1 and "val_coord_r2_relation" in last and "val_kd_loss" in last
