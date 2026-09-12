from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import networkx as nx
import pytest
import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator
from src.data.packed_features import PackedFeatureManifest, PackedFeatureTable, PackedNodeRecord
from src.data.struct_sampler import StructSampler
from src.distill.struct_config import StructConfig
from src.distill.struct_losses import struct_total
from src.train_b0 import (
    StructStream,
    ValidationOutcome,
    _grad_norm,
    config_to_dict,
    load_config,
    train_ddp_loop,
)
from torch import nn
from torch.multiprocessing.spawn import spawn

from tests.test_train_b0 import (
    _batch_of,
    _constant_metrics,
    _make_synthetic_pair_dataset,
    _tiny_config,
    _TinyPairMLP,
)


def _control_yaml() -> dict[str, object]:
    raw = yaml.safe_load(Path("configs/b1_kd_control_breadth_first.yaml").read_text())
    return dict(raw)


def test_struct_block_parses_and_serialises(tmp_path: Path) -> None:
    raw = _control_yaml()
    raw["struct"] = {"weights": {"bce": 1.0, "rank": 1.0, "degree": 0.1, "motif": 0.1}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    assert cfg.struct is not None
    assert cfg.struct.arm == "bce+degree+motif+rank"
    payload = config_to_dict(cfg)
    assert payload["struct"]["weights"]["rank"] == 1.0
    assert payload["struct"]["mix"] == {"bfs": 0.5, "motif": 0.25, "bridge": 0.25}


def test_absent_struct_block_is_none(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(_control_yaml()))
    assert load_config(path).struct is None


def test_struct_block_rejects_unknown_key(tmp_path: Path) -> None:
    raw = _control_yaml()
    raw["struct"] = {"weights": {"bce": 1.0}, "gradnorm": True}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown struct config keys"):
        load_config(path)


# --------------------------------------------------------------------------- StructStream


class _StructToy(nn.Module):
    """Logit = w * (sum emb_a - sum emb_b)^2 - 1 so every pair depends on one weight."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.forward_calls = 0
        self.batch_sizes: list[int] = []

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        self.batch_sizes.append(int(batch["emb_a"].shape[0]))
        a = batch["emb_a"].sum(dim=(1, 2))
        b = batch["emb_b"].sum(dim=(1, 2))
        return {"logits": self.weight * (a - b).square() - 1.0}


def _struct_fixture(n_nodes: int = 12) -> tuple[StructSampler, PackedFeatureTable]:
    graph = nx.relabel_nodes(
        nx.barabasi_albert_graph(n_nodes, 2, seed=1), {i: f"n{i}" for i in range(n_nodes)}
    )
    sampler = StructSampler(
        graph,
        nodes=8,
        background_nodes=2,
        mix={"bfs": 0.5, "motif": 0.25, "bridge": 0.25},
        v_val=frozenset({"n0", "n1"}),
        exclude_nodes=frozenset(),
    )
    node_ids = [f"n{i}" for i in range(n_nodes)]
    lengths = [1 + (i % 3) for i in range(n_nodes)]  # three length buckets
    records = tuple(
        PackedNodeRecord(node, 0, sum(lengths[:i]), sum(lengths[:i]), lengths[i])
        for i, node in enumerate(node_ids)
    )
    manifest = PackedFeatureManifest(
        format="test",
        input_dim=1,
        dtype="bfloat16",
        source_metadata_sha256="",
        source_index_sha256="",
        nodes=records,
        shards=(),
        pack_workers=1,
        build_seconds=0.0,
    )
    tokens = torch.arange(1, sum(lengths) + 1, dtype=torch.float32).unsqueeze(-1)
    offsets = torch.tensor([sum(lengths[:i]) for i in range(n_nodes)])
    table = PackedFeatureTable(tokens, offsets, torch.tensor(lengths), manifest)
    return sampler, table


def _stream(
    sampler: StructSampler,
    table: PackedFeatureTable,
    *,
    rank: int = 0,
    world_size: int = 1,
    weights: dict[str, float] | None = None,
    token_budget: int = 1 << 20,
    val_sampler: StructSampler | None = None,
) -> StructStream:
    config = StructConfig.from_mapping(
        {
            "nodes": 8,
            "background_nodes": 2,
            "val_subgraphs": 4,
            "weights": weights or {"bce": 1.0, "rank": 1.0, "degree": 0.1, "motif": 0.1},
        }
    )
    return StructStream(
        config,
        sampler,
        table,
        rank=rank,
        world_size=world_size,
        token_budget=token_budget,
        positive_weight=5.0,
        label_smoothing=0.0,
        seed=0,
        val_sampler=val_sampler,
    )


def test_stream_assembles_the_same_logits_as_a_direct_forward() -> None:
    sampler, table = _struct_fixture()
    stream = _stream(sampler, table, token_budget=3)  # forces many small chunks
    model = _StructToy()
    loss, stats = stream.loss(model, epoch=1, step=0, steps=4)
    subgraph = stream.last_subgraph
    assert subgraph is not None
    # Every stream chunk respects the budget of 3 tokens (one pair at boundary 3).
    assert max(model.batch_sizes) <= 3
    index = table.manifest.node_index()
    mask = torch.from_numpy(sampler.legal_mask(subgraph))
    target = torch.from_numpy(sampler.adjacency(subgraph))
    n = len(subgraph.nodes)
    rows, cols = torch.triu_indices(n, n, offset=1)
    keep = mask[rows, cols] > 0
    rows, cols = rows[keep], cols[keep]
    boundary = max(table.manifest.nodes[index[node]].length for node in subgraph.nodes)
    emb_a, len_a = table.gather_nodes(
        torch.tensor([index[subgraph.nodes[i]] for i in rows.tolist()]), boundary
    )
    emb_b, len_b = table.gather_nodes(
        torch.tensor([index[subgraph.nodes[j]] for j in cols.tolist()]), boundary
    )
    direct = model({"emb_a": emb_a, "emb_b": emb_b, "len_a": len_a, "len_b": len_b})["logits"]
    logits = torch.zeros(n, n)
    logits[rows, cols] = direct
    logits[cols, rows] = direct
    expected, _ = struct_total(
        logits, target, mask, stream.config, positive_weight=5.0, label_smoothing=0.0
    )
    torch.testing.assert_close(loss, expected)
    assert stats["struct_pairs"] == float(len(rows))


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_ddp_mean_loss_and_gradient_match_single_rank_at_every_step(world_size: int) -> None:
    """Changing world size preserves the subgraph, loss and gradient at every step."""
    sampler, table = _struct_fixture()
    steps = 3
    model = _StructToy()
    single = _stream(sampler, table, rank=0, world_size=1)
    ranks = [_stream(sampler, table, rank=r, world_size=world_size) for r in range(world_size)]
    for step in range(steps):
        expected = single.loss(model, epoch=2, step=step, steps=steps)[0]
        ddp_mean = torch.stack(
            [s.loss(model, epoch=2, step=step, steps=steps)[0] for s in ranks]
        ).mean()
        live = [
            position
            for r in range(world_size)
            for position in ranks[r]._positions(steps, rank=r, steps=steps, step=step)
        ]
        assert live == [step]
        torch.testing.assert_close(ddp_mean, expected)
        torch.testing.assert_close(
            torch.autograd.grad(ddp_mean, model.weight)[0],
            torch.autograd.grad(expected, model.weight)[0],
        )


@pytest.mark.parametrize("count", [1, 3, 68, 271, 272])
def test_four_rank_plan_is_spread_globally_once(count: int) -> None:
    sampler, table = _struct_fixture()
    stream = _stream(sampler, table, world_size=4)
    steps = 272
    schedule = [
        [
            position
            for rank in range(4)
            for position in stream._positions(count, rank=rank, steps=steps, step=step)
        ]
        for step in range(steps)
    ]
    assert [p for positions in schedule for p in positions] == list(range(count))
    assert all(len(positions) <= 1 for positions in schedule)
    live_steps = [step for step, positions in enumerate(schedule) if positions]
    assert live_steps == [position * steps // count for position in range(count)]
    if count == steps:
        assert all(len(positions) == 1 for positions in schedule)
        assert [
            sum(len(stream._positions(count, rank=r, steps=steps, step=t)) for t in range(steps))
            for r in range(4)
        ] == [68] * 4


def test_plan_longer_than_steps_is_rejected_and_shorter_plan_yields_zero_steps() -> None:
    sampler, table = _struct_fixture()
    kwargs: dict[str, object] = {
        "rank": 0,
        "world_size": 1,
        "token_budget": 1 << 20,
        "positive_weight": 5.0,
        "label_smoothing": 0.0,
        "seed": 0,
    }
    long_config = StructConfig.from_mapping(
        {"nodes": 8, "background_nodes": 2, "subgraphs_per_epoch": 9, "weights": {"bce": 1.0}}
    )
    stream = StructStream(long_config, sampler, table, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="subgraphs_per_epoch"):
        stream.loss(_StructToy(), epoch=1, step=0, steps=4)
    short_config = StructConfig.from_mapping(
        {"nodes": 8, "background_nodes": 2, "subgraphs_per_epoch": 1, "weights": {"bce": 1.0}}
    )
    stream = StructStream(short_config, sampler, table, **kwargs)  # type: ignore[arg-type]
    model = _StructToy()
    _, first = stream.loss(model, epoch=1, step=0, steps=4)
    zero, second = stream.loss(model, epoch=1, step=3, steps=4)
    # The single subgraph lands on exactly one of the four steps; the others are zero.
    assert (first["struct_pairs"] > 0) != (second["struct_pairs"] > 0)
    if second["struct_pairs"] == 0.0:
        assert zero.requires_grad and float(zero) == 0.0


def test_validation_scores_the_gold_val_graph_with_the_val_sampler() -> None:
    # The training sampler masks every V_val-internal pair; a V_val-only diagnostic subgraph
    # must be masked and targeted by the validation sampler, or every val term collapses to 0.
    train_sampler, table = _struct_fixture()
    v_val = frozenset(f"n{i}" for i in range(8))
    train_sampler = StructSampler(
        train_sampler.graph,
        nodes=8,
        background_nodes=2,
        mix={"bfs": 0.5, "motif": 0.25, "bridge": 0.25},
        v_val=v_val,
        exclude_nodes=frozenset(),
    )
    val_graph = train_sampler.graph.subgraph(v_val).copy()
    val_sampler = StructSampler(
        val_graph,
        nodes=8,
        background_nodes=0,
        mix={"bfs": 1.0, "motif": 0.0, "bridge": 0.0},
        v_val=frozenset(),
        exclude_nodes=frozenset(),
    )
    stream = _stream(train_sampler, table, weights={"bce": 1.0}, val_sampler=val_sampler)
    val = stream.validation_telemetry(_StructToy(), Accelerator(cpu=True), threshold=0.0)
    assert val["val_struct_bce_loss"] > 0.0
    assert val["val_struct_hard_degree_mae"] > 0.0


def test_epoch_and_validation_telemetry_keys() -> None:
    sampler, table = _struct_fixture()
    stream = _stream(sampler, table, val_sampler=sampler)
    model = _StructToy()
    accelerator = Accelerator(cpu=True)
    sums: dict[str, float] = {}
    for step in range(2):
        _, stats = stream.loss(model, epoch=1, step=step, steps=2)
        for key, value in stats.items():
            sums[key] = sums.get(key, 0.0) + value
    telemetry = stream.epoch_telemetry(accelerator, sums)
    for key in (
        "struct_bce_loss",
        "struct_rank_loss",
        "struct_degree_loss",
        "struct_motif_loss",
        "struct_pairs",
        "struct_positive_coverage",
        "struct_positive_reuse",
        "struct_components",
        "struct_legal_fraction",
        "struct_kind_bfs",
    ):
        assert key in telemetry, key
    assert telemetry["struct_pairs"] == sums["struct_pairs"]
    val = stream.validation_telemetry(model, accelerator, threshold=0.0)
    for key in (
        "val_struct_bce_loss",
        "val_struct_motif_loss",
        "val_struct_hard_degree_mae",
        "val_struct_hard_triangle_mae",
        "val_struct_hard_wedge_mae",
    ):
        assert key in val, key
    assert model.training  # validation restores the mode it found
    assert stream.last_terms.keys() == {"bce", "rank", "degree", "motif"}
    assert _grad_norm(stream.last_terms["bce"], model) > 0.0
    assert _grad_norm(torch.tensor(1.0), model) == 0.0


# --------------------------------------------------------------------------- loop wiring


def _task_batch() -> dict[str, torch.Tensor]:
    batch = _batch_of(_make_synthetic_pair_dataset(8, input_dim=4, seed=1))
    batch["_row_id"] = torch.arange(8)
    batch["_local_pair_count"] = torch.tensor(8)
    batch["_global_pair_count"] = torch.tensor(8)
    return batch


class _MLPOnTokens(_TinyPairMLP):
    """Serve the task batch and the stream's packed token batches.

    Task batches carry `x_a`/`x_b`; stream batches carry (B, L, 1) `emb_a`/`emb_b`
    tokens, mean-pooled and widened to `input_dim`.
    """

    def forward(
        self, batch: dict[str, torch.Tensor] | None = None, **kwargs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        merged = dict(batch or {})
        merged.update(kwargs)
        if "emb_a" in merged:
            merged = {
                "x_a": merged["emb_a"].mean(dim=1).expand(-1, 4),
                "x_b": merged["emb_b"].mean(dim=1).expand(-1, 4),
            }
        return super().forward(merged)


def _run_struct_ddp_fixture(
    path: Path, *, rank: int, world_size: int, count: int | None
) -> dict[str, torch.Tensor]:
    sampler, table = _struct_fixture()
    struct_cfg = StructConfig.from_mapping(
        {
            "nodes": 8,
            "background_nodes": 2,
            "val_subgraphs": 0,
            "subgraphs_per_epoch": count,
            "weights": {"bce": 1.0, "motif": 0.1},
        }
    )
    cfg = replace(_tiny_config(epochs=1), struct=struct_cfg)
    torch.manual_seed(7)
    model = _MLPOnTokens(input_dim=4, hidden_dims=(8,), dropout=0.0)
    stream = StructStream(
        struct_cfg,
        sampler,
        table,
        rank=rank,
        world_size=world_size,
        token_budget=1 << 20,
        positive_weight=5.0,
        label_smoothing=0.0,
        seed=0,
    )
    batches = []
    local_count = 16 // world_size
    for step in range(3):
        batch = {
            key: torch.cat([value, value])[rank * local_count : (rank + 1) * local_count]
            for key, value in _task_batch().items()
            if value.ndim > 0
        }
        batch["_row_id"] = torch.arange(local_count) + step * 16 + rank * local_count
        batch["_local_pair_count"] = torch.tensor(local_count)
        batch["_global_pair_count"] = torch.tensor(16)
        batches.append(batch)
    result = train_ddp_loop(
        model,
        lambda epoch: batches,
        batches,
        cfg,
        Accelerator(cpu=True),
        warmup_steps=1,
        artifact_dir=path,
        evaluate_fn=lambda model, loader, accelerator: ValidationOutcome(_constant_metrics(), None),
        struct_stream=stream,
    )
    # The returned checkpoint is main-rank-only; inspect live parameters on every rank.
    assert result.history
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _struct_ddp_worker(
    rank: int, world_size: int, init_file: str, result_dir: str, count: int | None
) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29500",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=45),
    )
    try:
        state = _run_struct_ddp_fixture(
            Path(result_dir) / "ddp", rank=rank, world_size=world_size, count=count
        )
        torch.save(state, Path(result_dir) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("count", [None, 1])
def test_real_ddp_struct_training_and_sparse_telemetry_match_serial(
    tmp_path: Path, count: int | None
) -> None:
    # Default: rank 1 first sees structure at step 1. Sparse: rank 1 never sees it.
    spawn(  # type: ignore[no-untyped-call]
        _struct_ddp_worker,
        args=(2, str(tmp_path / "init"), str(tmp_path), count),
        nprocs=2,
        join=True,
    )
    expected = _run_struct_ddp_fixture(tmp_path / "serial", rank=0, world_size=1, count=count)
    for rank in range(2):
        observed = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        for key in expected:
            torch.testing.assert_close(observed[key], expected[key], rtol=1e-5, atol=1e-6)
    ddp_row = json.loads((tmp_path / "ddp" / "metrics.jsonl").read_text().splitlines()[-1])
    serial_row = json.loads((tmp_path / "serial" / "metrics.jsonl").read_text().splitlines()[-1])
    for key in ("train_struct_loss", "struct_bce_loss", "struct_motif_loss", "struct_pairs"):
        assert ddp_row[key] == pytest.approx(serial_row[key], rel=1e-5)
    for key in ("grad_norm_struct_bce", "grad_norm_struct_motif"):
        assert ddp_row[key] > 0
        if count == 1:
            assert ddp_row[key] == pytest.approx(serial_row[key], rel=1e-5)
    weighted_raw = ddp_row["struct_bce_loss"] + 0.1 * ddp_row["struct_motif_loss"]
    assert ddp_row["train_struct_loss"] == pytest.approx(
        weighted_raw * (1.0 if count is None else count / 3), rel=1e-5
    )


def test_struct_stream_changes_weights_and_logs_keys(tmp_path: Path) -> None:
    sampler, table = _struct_fixture()
    struct_cfg = StructConfig.from_mapping(
        {
            "nodes": 8,
            "background_nodes": 2,
            "val_subgraphs": 2,
            "weights": {"bce": 1.0, "motif": 0.1},
        }
    )
    cfg = replace(_tiny_config(epochs=2), struct=struct_cfg)

    def _run(subdir: str, with_stream: bool) -> dict[str, torch.Tensor]:
        torch.manual_seed(7)
        model = _MLPOnTokens(input_dim=4, hidden_dims=(8,), dropout=0.0)
        stream = (
            StructStream(
                struct_cfg,
                sampler,
                table,
                rank=0,
                world_size=1,
                token_budget=1 << 20,
                positive_weight=5.0,
                label_smoothing=0.0,
                seed=0,
                val_sampler=sampler,
            )
            if with_stream
            else None
        )
        batch = _task_batch()
        result = train_ddp_loop(
            model,
            lambda epoch: [batch],
            [batch],
            cfg,
            Accelerator(cpu=True),
            warmup_steps=1,
            artifact_dir=tmp_path / subdir,
            evaluate_fn=lambda model, loader, accelerator: ValidationOutcome(
                _constant_metrics(), None
            ),
            struct_stream=stream,
        )
        return result.last_state_dict

    with_stream = _run("struct", True)
    without = _run("plain", False)
    assert any(not torch.equal(with_stream[key], without[key]) for key in with_stream), (
        "the structural loss must move the weights"
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "struct" / "metrics.jsonl").read_text().splitlines()
    ]
    last = rows[-1]
    for key in (
        "train_struct_loss",
        "struct_bce_loss",
        "struct_motif_loss",
        "grad_norm_struct_bce",
        "grad_norm_struct_motif",
        "struct_pairs",
        "struct_seconds",
        "struct_wall_fraction",
        "struct_components",
        "struct_positive_coverage",
        "val_struct_bce_loss",
    ):
        assert key in last, key
    plain_rows = [
        json.loads(line) for line in (tmp_path / "plain" / "metrics.jsonl").read_text().splitlines()
    ]
    assert not any(k.startswith(("struct_", "val_struct_", "train_struct")) for k in plain_rows[-1])
