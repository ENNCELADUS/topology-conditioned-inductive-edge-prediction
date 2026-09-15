"""Virtual student's combined task and structural objective under real DDP."""

from __future__ import annotations

import os
import socket
from datetime import timedelta
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
from src.model.egostitch.classifier.coord_gen import V3_1CoordGen
from src.train_b0 import TopoPromptRows
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel as DDP

from tests.test_prefix_model import _tiny_base_config
from tests.test_train_b0_struct import _stream, _topo_prompt_fixture


def _virtual_steps(rank: int, world: int) -> dict[str, torch.Tensor]:
    sampler, table, _ = _topo_prompt_fixture()
    pairs = [("n0", "n1"), ("n2", "n3"), ("n4", "n5"), ("n6", "n7")]
    rows = TopoPromptRows(
        train_graph=sampler.graph,
        train_pairs=pairs,
        stats_rows=torch.arange(len(pairs)).numpy(),
        val_graph=sampler.graph,
        val_cls_pairs=[],
        universe_pairs=[],
        device=torch.device("cpu"),
        spec="v2",
    )
    base = _tiny_base_config()
    base["regularization"] = dict.fromkeys(cast(dict[str, object], base["regularization"]), 0.0)
    base["mlp_head"] = {**cast(dict[str, object], base["mlp_head"]), "dropout": 0.0}
    torch.manual_seed(43)
    raw = V3_1CoordGen(
        reader={
            "base": base,
            "topo_prompt": {
                "coord_spec": "v2",
                "trainable": "all",
                "width": 8,
                "slots_per_field": 1,
                "field_mask_prob": 0.0,
            },
        },
        coord_gen={
            "coord_spec": "v2",
            "generator": "virtual_graph",
            "w_anchor": 1.0,
            "virtual_graph": {"k": 4, "d_z": 8, "heads": 2},
        },
    )
    rows.install(raw.reader)
    raw.install_coordinate_scale(rows.train)
    with torch.no_grad():
        raw.reader.generator.gates.fill_(0.3)
    raw.initialize_teacher()
    raw.train()
    model = DDP(raw, broadcast_buffers=False, bucket_cap_mb=0.001) if world > 1 else raw
    stream = _stream(sampler, table, rank=rank, world_size=world, token_budget=64, coordinates=rows)
    index = table.manifest.node_index()
    a, la = table.gather_nodes(torch.tensor([index[u] for u, _ in pairs]), 5)
    b, lb = table.gather_nodes(torch.tensor([index[v] for _, v in pairs]), 5)
    batch = {
        "emb_a": a,
        "emb_b": b,
        "len_a": la,
        "len_b": lb,
        "label": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "struct_coords": rows.train,
    }
    # Both local slices have identical positive-weight sums, so DDP's mean is
    # the full batch's weighted objective.
    local = {
        key: value[rank * (len(pairs) // world) : (rank + 1) * (len(pairs) // world)]
        for key, value in batch.items()
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    evidence: dict[str, torch.Tensor] = {}
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(local)
        structural, _ = stream.loss(model, epoch=1, step=step, steps=3)
        (output["loss"] + structural).backward()
        assert set(stream.last_terms) == {"bce", "rank", "degree", "motif", "anchor"}
        for name, parameter in raw.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                evidence[f"{step}/grad/{name}"] = parameter.grad.detach().clone()
        for name, loss in stream.last_terms.items():
            evidence[f"{step}/term/{name}"] = loss.detach().clone()
        assert output["coord_loss"] > 0
        optimizer.step()
    evidence.update(
        {f"state/{name}": value.detach().clone() for name, value in raw.state_dict().items()}
    )
    return evidence


def _virtual_worker(rank: int, root: str) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = (
        "lo0" if "lo0" in {name for _, name in socket.if_nameindex()} else "lo"
    )
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/init",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        torch.save(_virtual_steps(rank, 2), Path(root) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_virtual_combined_objective_two_rank_ddp_matches_serial(tmp_path: Path) -> None:
    spawn(_virtual_worker, args=(str(tmp_path),), nprocs=2, join=True)  # type: ignore[no-untyped-call]
    reference = _virtual_steps(0, 1)
    for rank in range(2):
        actual = torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
        assert actual.keys() == reference.keys()
        for name, expected in reference.items():
            torch.testing.assert_close(actual[name], expected, rtol=2e-4, atol=2e-6, msg=name)
