"""Two-stage optimizer, coordinate stream, and distributed objective contracts."""

from __future__ import annotations

import os
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.distributed as dist
from src.distill.struct_config import StructConfig
from src.model.egostitch.classifier.coord_gen import V3_1CoordGen
from src.model.egostitch.classifier.topo_prompt import COORDS_KEY
from src.train_b0 import (
    StructStream,
    TopoPromptRows,
    _build_optimizer,
    _build_scheduler,
    load_config,
)
from torch import nn
from torch.multiprocessing.spawn import spawn

from tests.test_prefix_model import _tiny_base_config
from tests.test_train_b0 import _tiny_config, _write_yaml_config
from tests.test_train_b0_struct import _struct_fixture


def _fixture(
    rank: int = 0, world_size: int = 1
) -> tuple[V3_1CoordGen, StructStream, dict[str, torch.Tensor]]:
    sampler, table = _struct_fixture()
    n = len(table.manifest.nodes)
    table.manifest = replace(
        table.manifest,
        nodes=tuple(
            replace(record, length=3, shard_offset=i * 3, global_offset=i * 3)
            for i, record in enumerate(table.manifest.nodes)
        ),
    )
    table.tokens = torch.arange(1, n * 3 + 1, dtype=torch.float32).reshape(-1, 1)
    table.offsets = torch.arange(n) * 3
    table.lengths = torch.full((n,), 3)
    base = _tiny_base_config()
    base["input_dim"] = 1
    base["regularization"] = dict.fromkeys(cast(dict[str, object], base["regularization"]), 0.0)
    base["mlp_head"] = {**cast(dict[str, object], base["mlp_head"]), "dropout": 0.0}
    prompt = {"trainable": "all", "width": 8, "slots_per_field": 1, "field_mask_prob": 0.0}
    pairs = list(sampler.graph.edges())
    rows = TopoPromptRows(
        train_graph=sampler.graph,
        train_pairs=pairs,
        stats_rows=np.arange(len(pairs)),
        val_graph=sampler.graph,
        val_cls_pairs=pairs,
        universe_pairs=pairs,
        device=torch.device("cpu"),
    )
    torch.manual_seed(7)
    model = V3_1CoordGen(
        reader={"base": base, "topo_prompt": prompt},
        coord_gen={
            "hidden": 8,
            "endpoint_hidden": 8,
            "layers": 1,
            "dropout": 0.0,
            "endpoint_dropout": 0.0,
        },
    )
    rows.install(model.reader)
    with torch.no_grad():
        model.reader.generator.gates.fill_(0.3)
    model.initialize_teacher()
    config = StructConfig(
        nodes=8,
        background_nodes=2,
        val_subgraphs=0,
        weights={"bce": 1.0, "rank": 1.0, "degree": 0.1, "motif": 0.1},
    )
    stream = StructStream(
        config,
        sampler,
        table,
        rank=rank,
        world_size=world_size,
        token_budget=128,
        positive_weight=5.0,
        label_smoothing=0.0,
        seed=2,
        coordinates=rows,
    )
    index = table.manifest.node_index()
    a, b = zip(*pairs[:4], strict=True)
    ia = torch.tensor([index[node] for node in a])
    ib = torch.tensor([index[node] for node in b])
    ea, la = table.gather_nodes(ia, 3)
    eb, lb = table.gather_nodes(ib, 3)
    batch = {
        "emb_a": ea,
        "emb_b": eb,
        "len_a": la,
        "len_b": lb,
        "label": torch.tensor([1, 0, 1, 0]),
        COORDS_KEY: rows.train[:4],
    }
    return model, stream, batch


def test_stream_uses_full_table_and_teacher_and_backward() -> None:
    model, stream, _ = _fixture()
    teacher_before = {k: v.clone() for k, v in model.teacher.state_dict().items()}
    model.train()
    loss, stats = stream.loss(model, epoch=1, step=0, steps=2)
    assert stats["sum_struct_anchor_entropy"] > 0
    assert stats["sum_struct_anchor_loss"] >= 0
    assert "anchor" in stream.last_terms
    loss.backward()  # type: ignore[no-untyped-call]
    assert any(p.grad is not None for p in model.generator.parameters())
    assert any(p.grad is not None for p in model.reader.generator.parameters())
    assert all(p.grad is None for p in model.teacher.parameters())
    for key, value in teacher_before.items():
        torch.testing.assert_close(value, model.teacher.state_dict()[key], rtol=0, atol=0)
    # The same stream also accepts Stage I and obtains coordinates from the full table.
    reader = model.teacher
    reader.requires_grad_(True)
    stream.loss(reader, epoch=1, step=0, steps=2)[0].backward()  # type: ignore[no-untyped-call]


def test_groups_and_null_patience_parse_and_reach_onecycle(tmp_path: Path) -> None:
    path = tmp_path / "groups.yaml"
    _write_yaml_config(
        path,
        {
            "optim": {
                **asdict(_tiny_config().optim),
                "groups": {"generator": {"max_lr": 3e-4}, "interface": {"max_lr": 1e-4}},
                "scheduler": {
                    "type": "onecycle",
                    "max_lr": 3e-4,
                    "pct_start": 0.1,
                    "anneal_strategy": "cos",
                    "div_factor": 25,
                    "final_div_factor": 100,
                },
            },
            "eval": {**asdict(_tiny_config().eval), "patience": None},
        },
    )
    cfg = load_config(path)
    assert cfg.eval.patience is None
    model, _, _ = _fixture()
    optimizer = _build_optimizer(model, cfg)
    _build_scheduler(optimizer, cfg, warmup_steps=0, total_steps=100)
    assert {g["name"]: g["max_lr"] for g in optimizer.param_groups} == {
        "endpoint": 3e-4,
        "generator": 3e-4,
        "interface": 1e-4,
    }
    assert optimizer.param_groups[0]["weight_decay"] == model.cfg.endpoint_weight_decay


def _step(rank: int, world_size: int) -> dict[str, torch.Tensor]:
    model, stream, batch = _fixture(rank, world_size)
    model.train()
    wrapped = nn.parallel.DistributedDataParallel(model) if world_size > 1 else model
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for step in range(2):
        output = wrapped(batch)
        structural, _ = stream.loss(wrapped, epoch=1, step=step, steps=2)
        optimizer.zero_grad()
        (output["loss"] + structural).backward()
        optimizer.step()
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _worker(rank: int, root: str) -> None:
    torch.set_num_threads(1)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if os.uname().sysname == "Darwin" else "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/init",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        torch.save(_step(rank, 2), Path(root) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_real_two_rank_online_anchor_stream_matches_serial(tmp_path: Path) -> None:
    spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)  # type: ignore[no-untyped-call]
    expected = _step(0, 1)
    for rank in range(2):
        actual = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-4, atol=2e-6)


def test_null_patience_runs_entire_budget_despite_worsening_loss(tmp_path: Path) -> None:
    from accelerate import Accelerator
    from src.train_b0 import ValidationOutcome, train_ddp_loop

    from tests.test_train_b0 import _constant_metrics
    from tests.test_train_b0_struct import _MLPOnTokens, _task_batch

    cfg = _tiny_config(epochs=3)
    cfg = replace(cfg, eval=replace(cfg.eval, patience=None))
    counter = 0

    def evaluate(model: nn.Module, loader: object, accelerator: Accelerator) -> ValidationOutcome:
        nonlocal counter
        counter += 1
        return ValidationOutcome(_constant_metrics(), None, float(counter))

    batch = _task_batch()
    result = train_ddp_loop(
        _MLPOnTokens(input_dim=4, hidden_dims=(8,), dropout=0.0),
        lambda epoch: [batch],
        [batch],
        cfg,
        Accelerator(cpu=True),
        warmup_steps=1,
        artifact_dir=tmp_path,
        evaluate_fn=evaluate,
    )
    assert result.last_epoch == 3
    assert not result.stopped_early
    assert [row["val_task_loss"] for row in result.history] == [1, 2, 3]
