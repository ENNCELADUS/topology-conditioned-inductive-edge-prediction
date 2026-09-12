"""train_b0 dispatch, coordinate rows, and run-kind plumbing for the v3_1_topo_prompt family."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import pytest
import torch
from accelerate import Accelerator
from src.data.struct_coords import COORD_DIM
from src.eval.checkpoint_selection import TopologyValidationMetrics
from src.eval.val_topology import ValTopologyResult
from src.model.egostitch.classifier.b0_v31 import V3_1
from src.model.egostitch.classifier.topo_prompt import V3_1TopoPrompt
from src.train_b0 import (
    MODEL_FAMILIES,
    TopoPromptRows,
    TrainResult,
    ValidationOutcome,
    _base_loss_kwargs,
    _build_optimizer,
    _evaluate_distributed,
    _evaluate_two_pass,
    _run_metadata,
    apply_overrides,
    build_model,
    config_to_dict,
    is_v3_1_family,
    load_config,
    parse_args,
    resolve_model_kwargs,
    train_ddp_loop,
)

from tests.test_prefix_model import _pair_batch, _tiny_base_config
from tests.test_train_b0 import _constant_metrics, _tiny_config, _write_yaml_config
from tests.test_train_b0_prefix import _base_checkpoint


def _topo_yaml(tmp_path: Path, *, trainable: str = "all") -> Path:
    config_path = tmp_path / "topo.yaml"
    block: dict[str, object] = {"trainable": trainable, "width": 8, "slots_per_field": 1}
    model_config: dict[str, object] = {"topo_prompt": block}
    if trainable == "prompt":
        block["base_checkpoint"] = str(_base_checkpoint(tmp_path))
    else:
        model_config["base"] = _tiny_base_config()
    _write_yaml_config(
        config_path, {"model": {"family": "v3_1_topo_prompt", "config": model_config}}
    )
    return config_path


def test_family_is_registered_and_grouped_with_v3_1() -> None:
    assert "v3_1_topo_prompt" in MODEL_FAMILIES
    assert is_v3_1_family("v3_1_topo_prompt")


def test_resolve_and_build_from_an_explicit_base(tmp_path: Path) -> None:
    cfg = load_config(_topo_yaml(tmp_path))
    kwargs = resolve_model_kwargs(cfg.model)
    assert kwargs["base"] == _tiny_base_config()
    block = cast(dict[str, object], kwargs["topo_prompt"])
    assert block["trainable"] == "all" and block["base_checkpoint"] == ""
    assert _base_loss_kwargs(cfg.model)["positive_weight"] == 5.0
    model = build_model(cfg)
    assert isinstance(model, V3_1TopoPrompt) and not model.frozen_base
    optimizer = _build_optimizer(model, cfg)
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(
        list(model.parameters())
    )


def test_resolve_and_build_from_a_base_checkpoint(tmp_path: Path) -> None:
    cfg = load_config(_topo_yaml(tmp_path, trainable="prompt"))
    kwargs = resolve_model_kwargs(cfg.model)
    block = cast(dict[str, object], kwargs["topo_prompt"])
    assert block["base_checkpoint_sha256"] is not None
    model = build_model(cfg)
    assert isinstance(model, V3_1TopoPrompt) and model.frozen_base
    base = V3_1(**_tiny_base_config())
    torch.manual_seed(0)
    base = V3_1(**_tiny_base_config())
    for key, value in base.state_dict().items():
        torch.testing.assert_close(model.base.state_dict()[key], value)
    optimizer = _build_optimizer(model, cfg)
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(
        list(model.generator.parameters())
    )


def test_base_source_must_be_exactly_one(tmp_path: Path) -> None:
    both = tmp_path / "both.yaml"
    _write_yaml_config(
        both,
        {
            "model": {
                "family": "v3_1_topo_prompt",
                "config": {
                    "topo_prompt": {"base_checkpoint": str(_base_checkpoint(tmp_path))},
                    "base": _tiny_base_config(),
                },
            }
        },
    )
    with pytest.raises(ValueError, match="exactly one"):
        resolve_model_kwargs(load_config(both).model)
    neither = tmp_path / "neither.yaml"
    _write_yaml_config(
        neither, {"model": {"family": "v3_1_topo_prompt", "config": {"topo_prompt": {}}}}
    )
    with pytest.raises(ValueError, match="exactly one"):
        resolve_model_kwargs(load_config(neither).model)
    extra = tmp_path / "extra.yaml"
    _write_yaml_config(
        extra,
        {
            "model": {
                "family": "v3_1_topo_prompt",
                "config": {"topo_prompt": {}, "base": _tiny_base_config(), "prefix": {}},
            }
        },
    )
    with pytest.raises(ValueError, match="accepts only"):
        resolve_model_kwargs(load_config(extra).model)


def test_run_kind_flows_from_the_cli_and_stays_out_of_the_config_hash(tmp_path: Path) -> None:
    config_path = _topo_yaml(tmp_path)
    args = parse_args(["--config", str(config_path), "--run-kind", "diagnostic"])
    cfg = apply_overrides(load_config(config_path), args)
    assert cfg.run_kind == "diagnostic"
    assert "run_kind" not in config_to_dict(cfg)
    assert config_to_dict(cfg) == config_to_dict(replace(cfg, run_kind=None))
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config_path), "--run-kind", "overfit"])


def test_run_metadata_records_run_kind_and_prompt_provenance(tmp_path: Path) -> None:
    cfg = replace(load_config(_topo_yaml(tmp_path)), run_kind="diagnostic")
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
    assert metadata["run_kind"] == "diagnostic"
    assert metadata["checkpoint_role"] == "diagnostic_only"
    assert metadata["formal_artifacts_published"] is False
    prompt = cast(dict[str, object], metadata["topo_prompt"])
    assert prompt["trainable"] == "all" and prompt["base_checkpoint"] is None
    json.dumps(metadata)


def _tiny_graph() -> tuple[nx.Graph, nx.Graph, list[str]]:
    train = nx.relabel_nodes(
        nx.barabasi_albert_graph(14, 2, seed=3), {i: f"n{i}" for i in range(14)}
    )
    val = nx.relabel_nodes(nx.cycle_graph(5), {i: f"v{i}" for i in range(5)})
    return train, val, sorted(train.nodes)


def test_rows_measure_every_universe_and_attach_by_row_id() -> None:
    train, val, nodes = _tiny_graph()
    train_pairs = [(nodes[0], nodes[1]), (nodes[2], nodes[5]), (nodes[3], nodes[3])]
    val_pairs = [("v0", "v1"), ("v0", "v2")]
    universe = [("v1", "v3")]
    rows = TopoPromptRows(
        train_graph=train,
        train_pairs=train_pairs,
        stats_rows=np.asarray([0, 1, 2]),
        val_graph=val,
        val_cls_pairs=val_pairs,
        universe_pairs=universe,
        device=torch.device("cpu"),
    )
    assert rows.train.shape == (3, COORD_DIM)
    assert rows.val_cls.shape == (2, COORD_DIM)
    assert rows.universe.shape == (1, COORD_DIM)
    assert rows.coord_count == 3 and rows.coord_std.min() > 0
    batch = {"_row_id": torch.tensor([2, 0])}
    rows.attach_train(batch)
    torch.testing.assert_close(batch["struct_coords"], rows.train[[2, 0]])
    val_batch = {"_row_id": torch.tensor([1])}
    rows.attach_val(val_batch)
    torch.testing.assert_close(val_batch["struct_coords"], rows.val_cls[[1]])
    model = V3_1TopoPrompt(base=_tiny_base_config(), topo_prompt={"trainable": "all", "width": 8})
    rows.install(model)
    assert float(model.generator.coord_count) == 3.0
    assert rows.summary()["train_rows"] == 3


def _prompt_batches(n_batches: int, rows_per_batch: int) -> list[dict[str, torch.Tensor]]:
    batches = []
    for step in range(n_batches):
        batch = {
            key: value[:rows_per_batch]
            for key, value in _pair_batch(rows_per_batch, seed=step).items()
        }
        batch["_row_id"] = torch.arange(rows_per_batch) + step * rows_per_batch
        batch["_local_pair_count"] = torch.tensor(rows_per_batch)
        batch["_global_pair_count"] = torch.tensor(rows_per_batch)
        batches.append(batch)
    return batches


def test_ddp_loop_trains_the_prompt_model_from_attached_coordinates(tmp_path: Path) -> None:
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
    torch.manual_seed(0)
    model = V3_1TopoPrompt(
        base=_tiny_base_config(),
        topo_prompt={"trainable": "all", "width": 8, "slots_per_field": 1},
    )
    rows.install(model)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    batches = _prompt_batches(n_batches, rows_per_batch)
    val_batches = _prompt_batches(1, len(val_pairs))
    cfg = replace(_tiny_config(epochs=1), output_dir=tmp_path)
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
    assert result.history and result.history[-1]["epoch"] == 1
    after = model.state_dict()
    assert any(
        not torch.equal(before[key], after[key])
        for key in before
        if key.startswith("base.") and before[key].dtype.is_floating_point
    )
    assert not torch.equal(before["generator.gates"], after["generator.gates"])
    torch.testing.assert_close(after["generator.coord_mean"], rows.coord_mean)


def test_two_pass_validation_forwards_the_coordinate_hook() -> None:
    train, val, nodes = _tiny_graph()
    val_pairs = [("v0", "v1"), ("v1", "v3"), ("v2", "v4"), ("v0", "v3")]
    rows = TopoPromptRows(
        train_graph=train,
        train_pairs=[(nodes[0], nodes[1]), (nodes[2], nodes[3])],
        stats_rows=np.arange(2),
        val_graph=val,
        val_cls_pairs=val_pairs,
        universe_pairs=val_pairs,
        device=torch.device("cpu"),
    )
    torch.manual_seed(0)
    model = V3_1TopoPrompt(base=_tiny_base_config(), topo_prompt={"trainable": "all", "width": 8})
    rows.install(model)
    val_batches = _prompt_batches(1, len(val_pairs))
    topology = ValTopologyResult(
        metrics=TopologyValidationMetrics(
            gs=0.5, rd=1.0, degree_mmd=1.0, clustering_mmd=1.0, spectral_mmd=1.0
        ),
        threshold=0.0,
    )
    accelerator = Accelerator(cpu=True)
    outcome = _evaluate_two_pass(
        model,
        val_batches,
        accelerator,
        expected_row_ids=np.arange(len(val_pairs)),
        topology_eval_fn=lambda m, a: topology,
        attach=rows.attach_val,
    )
    assert outcome.topology is topology and outcome.task_loss is not None
    with pytest.raises(ValueError, match="struct_coords"):
        _evaluate_two_pass(
            model,
            _prompt_batches(1, len(val_pairs)),
            accelerator,
            expected_row_ids=np.arange(len(val_pairs)),
            topology_eval_fn=lambda m, a: topology,
        )


def test_ddp_loop_fails_closed_without_coordinates(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = V3_1TopoPrompt(base=_tiny_base_config(), topo_prompt={"trainable": "all", "width": 8})
    model.generator.set_coord_stats(torch.zeros(COORD_DIM), torch.ones(COORD_DIM), 1)
    cfg = replace(_tiny_config(epochs=1), output_dir=tmp_path)
    with pytest.raises(ValueError, match="struct_coords"):
        train_ddp_loop(
            model,
            lambda epoch: _prompt_batches(1, 4),
            _prompt_batches(1, 4),
            cfg,
            Accelerator(cpu=True),
            warmup_steps=1,
            artifact_dir=tmp_path,
            evaluate_fn=lambda m, loader, acc: ValidationOutcome(_constant_metrics(), None),
        )
